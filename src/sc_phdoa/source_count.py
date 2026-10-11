"""Source-count feature extraction and frozen classifier interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np

from .config import DEFAULT_CONFIG, ModelConfig


def normalized_covariance_eigenvalues(snapshots: np.ndarray) -> np.ndarray:
    snapshots = np.asarray(snapshots, np.complex128)
    centered = snapshots - snapshots.mean(axis=1, keepdims=True)
    covariance = centered @ centered.conj().T / max(centered.shape[1], 1)
    values = np.maximum(np.linalg.eigvalsh(covariance).real[::-1], 1e-12)
    return values / max(float(values.sum()), 1e-12)


def _append_eigen_features(
    values: np.ndarray,
    prefix: str,
    names: list[str],
    output: list[float],
) -> None:
    values = np.maximum(np.asarray(values, np.float64), 1e-12)
    normalized = values / max(float(values.sum()), 1e-12)
    log_values = np.log(normalized)
    gaps = np.log(values[:-1] / values[1:])
    entropy = -float(np.sum(normalized * np.log(normalized)))
    for index, value in enumerate(log_values):
        names.append(f"{prefix}_log_eig{index + 1}")
        output.append(float(value))
    for index, value in enumerate(gaps):
        names.append(f"{prefix}_gap{index + 1}{index + 2}")
        output.append(float(value))
    for count in (1, 2, 3):
        names.append(f"{prefix}_energy_top{count}")
        output.append(float(normalized[:count].sum()))
    names.extend([f"{prefix}_effective_rank", f"{prefix}_stable_rank"])
    output.extend(
        [float(np.exp(entropy)), float(values.sum() / max(values[0], 1e-12))]
    )


def _append_block_stability_features(
    payload: np.ndarray,
    temporal_blocks: int,
    names: list[str],
    output: list[float],
) -> None:
    blocks = np.array_split(payload, temporal_blocks, axis=1)
    values = np.stack([normalized_covariance_eigenvalues(block) for block in blocks])
    log_values = np.log(np.maximum(values, 1e-12))
    gaps = np.log(np.maximum(values[:, :-1] / np.maximum(values[:, 1:], 1e-12), 1e-12))
    groups = (
        (log_values.mean(axis=0), "block_logeig{}_mean"),
        (log_values.std(axis=0), "block_logeig{}_std"),
        (gaps.mean(axis=0), "block_gap{}{}_mean"),
        (gaps.std(axis=0), "block_gap{}{}_std"),
    )
    for values_group, template in groups:
        for index, value in enumerate(values_group):
            if "gap" in template:
                names.append(template.format(index + 1, index + 2))
            else:
                names.append(template.format(index + 1))
            output.append(float(value))


def ambiguity_surface(
    snapshots: np.ndarray,
    preamble: np.ndarray,
    max_delay: int,
    config: ModelConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    length = len(preamble)
    if snapshots.shape[1] < max_delay + length:
        raise ValueError("Snapshot length is shorter than delay plus preamble")
    windows = np.lib.stride_tricks.sliding_window_view(
        snapshots, length, axis=1
    )[:, : max_delay + 1, :]
    products = windows * np.conj(preamble)[None, None, :]
    spectrum = np.fft.fft(products, n=config.sync_fft_size, axis=-1)
    frequencies = np.fft.fftfreq(config.sync_fft_size)
    low, high = config.cfo_search_bounds
    keep = (frequencies >= low) & (frequencies <= high)
    return spectrum[:, :, keep], frequencies[keep]


def best_unexcluded_peak(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    selected: list[tuple[int, float]],
    config: ModelConfig = DEFAULT_CONFIG,
) -> tuple[int, float]:
    score = np.sum(np.abs(spectrum) ** 2, axis=0)
    delay_axis = np.arange(score.shape[0])[:, None]
    for delay, cfo in selected:
        suppress = (
            (np.abs(delay_axis - delay) <= config.nms_delay)
            & (np.abs(frequencies[None, :] - cfo) <= config.nms_cfo)
        )
        score[suppress] = -np.inf
    flat = int(np.argmax(score))
    delay_index, frequency_index = np.unravel_index(flat, score.shape)
    frequency = float(frequencies[frequency_index])
    if 0 < frequency_index < score.shape[1] - 1:
        values = np.log(
            np.maximum(score[delay_index, frequency_index - 1 : frequency_index + 2], 1e-20)
        )
        left, center, right = values
        denominator = left - 2.0 * center + right
        if np.isfinite(denominator) and abs(denominator) > 1e-12:
            offset = np.clip(0.5 * (left - right) / denominator, -0.5, 0.5)
            frequency += float(offset) * float(frequencies[1] - frequencies[0])
    return int(delay_index), frequency


def _append_sync_features(
    snapshots: np.ndarray,
    preamble: np.ndarray,
    max_delay: int,
    names: list[str],
    output: list[float],
    config: ModelConfig,
) -> None:
    spectrum, frequencies = ambiguity_surface(snapshots, preamble, max_delay, config)
    score = np.sum(np.abs(spectrum) ** 2, axis=0)
    selected: list[tuple[int, float]] = []
    peak_scores: list[float] = []
    signatures: list[np.ndarray] = []
    for _ in range(3):
        delay, cfo = best_unexcluded_peak(spectrum, frequencies, selected, config)
        selected.append((delay, cfo))
        frequency_index = int(np.argmin(np.abs(frequencies - cfo)))
        peak_scores.append(float(max(score[delay, frequency_index], 1e-20)))
        signatures.append(spectrum[:, delay, frequency_index])

    peak_scores_array = np.asarray(peak_scores, np.float64)
    for index, value in enumerate(np.log(peak_scores_array / max(peak_scores_array[0], 1e-20))):
        names.append(f"sync_log_peak{index + 1}_to_peak1")
        output.append(float(value))
    names.extend(["sync_peak12_logratio", "sync_peak23_logratio"])
    output.extend(
        [
            float(np.log(peak_scores_array[0] / peak_scores_array[1])),
            float(np.log(peak_scores_array[1] / peak_scores_array[2])),
        ]
    )
    signature_matrix = np.stack(signatures, axis=1)
    signature_matrix /= np.maximum(np.linalg.norm(signature_matrix, axis=0, keepdims=True), 1e-12)
    singular = np.linalg.svd(signature_matrix, compute_uv=False)
    singular /= max(float(singular[0]), 1e-12)
    for index, value in enumerate(singular):
        names.append(f"sync_signature_sv{index + 1}")
        output.append(float(value))

    delays = np.asarray([item[0] for item in selected], np.float64)
    cfos = np.asarray([item[1] for item in selected], np.float64)
    delay_sep = min(abs(delays[i] - delays[j]) for i in range(3) for j in range(i + 1, 3))
    cfo_sep = min(abs(cfos[i] - cfos[j]) for i in range(3) for j in range(i + 1, 3))
    names.extend(["sync_min_delay_sep", "sync_min_cfo_sep"])
    output.extend(
        [
            float(delay_sep / max(max_delay, 1)),
            float(cfo_sep / max(config.cfo_search_bounds[1] - config.cfo_search_bounds[0], 1e-12)),
        ]
    )


def extract_source_count_features(
    snapshots: np.ndarray,
    payload_start: int,
    preamble: np.ndarray,
    max_delay: int,
    config: ModelConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, list[str]]:
    """Return the locked 79-dimensional analytical feature vector."""
    scale = max(float(np.sqrt(np.mean(np.abs(snapshots) ** 2))), 1e-8)
    snapshots = np.asarray(snapshots, np.complex128) / scale
    full_values = normalized_covariance_eigenvalues(snapshots)
    payload = snapshots[:, payload_start:]
    payload_values = normalized_covariance_eigenvalues(payload)
    names: list[str] = []
    output: list[float] = []
    _append_eigen_features(full_values, "full", names, output)
    _append_eigen_features(payload_values, "payload", names, output)
    cross = np.log(np.maximum(full_values, 1e-12)) - np.log(np.maximum(payload_values, 1e-12))
    for index, value in enumerate(cross):
        names.append(f"cross_logeig{index + 1}_full_minus_payload")
        output.append(float(value))
    _append_block_stability_features(payload, config.temporal_blocks, names, output)
    _append_sync_features(snapshots, preamble, max_delay, names, output, config)
    return np.asarray(output, np.float32), names


@dataclass
class SourceCountPrediction:
    count: int
    probabilities: np.ndarray


class SourceCountEstimator:
    """Thin typed interface around the frozen scikit-learn count head."""

    def __init__(self, bundle: dict[str, Any], config: ModelConfig = DEFAULT_CONFIG) -> None:
        required = {"model", "columns", "feature_names", "family"}
        missing = required.difference(bundle)
        if missing:
            raise KeyError(f"Source-count checkpoint is missing: {sorted(missing)}")
        self.model = bundle["model"]
        self.columns = np.asarray(bundle["columns"], np.int64)
        self.feature_names = [str(name) for name in bundle["feature_names"]]
        self.family = str(bundle["family"])
        self.config = config

    @classmethod
    def load(cls, path: Union[str, Path], config: ModelConfig = DEFAULT_CONFIG) -> "SourceCountEstimator":
        import joblib
        import sklearn

        checkpoint_sklearn_version = "1.6.1"
        if sklearn.__version__ != checkpoint_sklearn_version:
            raise RuntimeError(
                "The bundled source-count checkpoint was serialized with "
                f"scikit-learn {checkpoint_sklearn_version}, but the active environment has "
                f"scikit-learn {sklearn.__version__}. Install the pinned dependency with "
                "`python -m pip install -r requirements.txt`."
            )

        return cls(joblib.load(Path(path)), config)

    def predict(
        self,
        snapshots: np.ndarray,
        payload_start: int,
        preamble: np.ndarray,
        max_delay: int,
    ) -> SourceCountPrediction:
        features, names = extract_source_count_features(
            snapshots, payload_start, preamble, max_delay, self.config
        )
        if names != self.feature_names:
            raise ValueError("Generated source-count feature order differs from the checkpoint")
        selected = features[self.columns][None]
        count = int(self.model.predict(selected)[0])
        probabilities = np.zeros(self.config.max_sources, np.float32)
        if hasattr(self.model, "predict_proba"):
            raw = np.asarray(self.model.predict_proba(selected)[0], np.float64)
            classes = np.asarray(self.model.classes_, np.int64)
            probabilities[classes - 1] = raw
        else:
            probabilities[count - 1] = 1.0
        return SourceCountPrediction(count=count, probabilities=probabilities)
