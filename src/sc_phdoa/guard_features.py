"""Guard-scene projection residuals and the locked 190-D router input."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .manifold import angles_from_mixing, direction

MAXIMUM_GUARD_TOKENS = 12


def projection_score(mixing: np.ndarray, manifold: Callable[[np.ndarray], np.ndarray]) -> float:
    predicted = manifold(angles_from_mixing(mixing, manifold))
    values = []
    for source in range(mixing.shape[1]):
        observed, model = mixing[:, source], predicted[:, source]
        denominator = max(
            float(np.vdot(observed, observed).real * np.vdot(model, model).real), 1e-12
        )
        values.append(1.0 - abs(np.vdot(model, observed)) ** 2 / denominator)
    return float(np.mean(values))


def source_residual_token(
    observed: np.ndarray,
    predicted: np.ndarray,
    angle: np.ndarray,
    source_count: int,
    snr_db: float,
    gates: np.ndarray,
    calibration_norm: float,
) -> np.ndarray:
    observed = np.asarray(observed).reshape(7)
    predicted = np.asarray(predicted).reshape(7)
    observed_reference = observed / (observed[0] if abs(observed[0]) > 1e-8 else 1.0)
    predicted_reference = predicted / (predicted[0] if abs(predicted[0]) > 1e-8 else 1.0)
    relative = observed_reference[1:] / np.where(
        np.abs(predicted_reference[1:]) > 1e-8, predicted_reference[1:], 1.0
    ) - 1.0
    left = observed / max(float(np.linalg.norm(observed)), 1e-12)
    right = predicted / max(float(np.linalg.norm(predicted)), 1e-12)
    phase = np.vdot(right, left)
    if abs(phase) > 1e-10:
        right *= phase / abs(phase)
    residual = left - right
    coherence = float(np.clip(abs(np.vdot(right, left)) ** 2, 0.0, 1.0))
    token = np.concatenate(
        (
            np.clip(relative.real, -3.0, 3.0),
            np.clip(relative.imag, -3.0, 3.0),
            np.clip(residual.real, -1.5, 1.5),
            np.clip(residual.imag, -1.5, 1.5),
            np.clip(np.abs(relative), 0.0, 3.0),
            direction(np.asarray(angle)[None])[0],
            [source_count / 3.0, np.clip(snr_db / 20.0, -0.25, 1.0)],
            [coherence, 1.0 - coherence],
            np.asarray(gates).reshape(-1),
            [calibration_norm],
        )
    )
    return np.nan_to_num(token, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)


def aggregate_tokens(tokens: Sequence[np.ndarray]) -> np.ndarray:
    if not tokens:
        return np.zeros(130, np.float32)
    values = np.asarray(tokens[:MAXIMUM_GUARD_TOKENS], np.float64)
    return np.concatenate(
        (
            values.mean(axis=0),
            values.std(axis=0),
            np.max(np.abs(values), axis=0),
            [len(values) / MAXIMUM_GUARD_TOKENS],
        )
    ).astype(np.float32)


def build_guard_features(
    initial_calibration: np.ndarray,
    gates: np.ndarray,
    nominal_score: float,
    context_score: float,
    source_tokens: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.asarray(initial_calibration, np.float32).reshape(-1)
    gates = np.asarray(gates, np.float32).reshape(-1)
    calibration_norm = float(np.linalg.norm(initial) / np.sqrt(len(initial)))
    summary = np.asarray(
        [nominal_score, context_score, nominal_score - context_score, *gates, calibration_norm],
        np.float32,
    )
    residual_input = np.concatenate(
        [initial, gates, summary, aggregate_tokens(source_tokens)]
    ).astype(np.float32)
    if residual_input.shape != (190,):
        raise RuntimeError(f"unexpected guard residual feature shape: {residual_input.shape}")
    return summary, residual_input
