"""External-session preprocessing and evaluation for controlled SetNets."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from ..dataset import SessionDataset, read_doas
from ..metrics import matched_errors, ospa
from ..protocol import protocol_fingerprint
from .neural_set import load_setnet_checkpoint


def _scene_representation(snapshots: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    snapshots = np.asarray(snapshots, np.complex64)
    centered = snapshots - snapshots.mean(axis=1, keepdims=True)
    covariance = centered @ centered.conj().T / max(centered.shape[1], 1)
    power = max(float(np.trace(covariance).real / 7.0), 1e-10)
    normalized = covariance / power
    covariance_channels = np.stack([normalized.real, normalized.imag]).astype(np.float32)
    # The released training/evaluation cache stores raw IQ as float16. Keep
    # that deterministic quantization before converting tensors to float32.
    iq_channels = np.stack([snapshots.real, snapshots.imag]).astype(np.float16).astype(np.float32)
    return covariance_channels, iq_channels, float(np.log(power))


def _metric_row(dataset: SessionDataset, handle: h5py.File, state: int, scene: int, estimate: np.ndarray, method: str) -> dict:
    original = int(dataset.original_indices[state, scene])
    true_count = int(dataset.source_counts[state, scene])
    truth = read_doas(handle, original, dataset.sample_count, true_count)
    predicted_count = len(estimate)
    correct = predicted_count == true_count
    errors = matched_errors(estimate, truth) if correct else np.asarray([])
    maximum = float(np.max(errors)) if len(errors) else float("nan")
    return {
        "method": method,
        "state_index": state,
        "state_id": int(dataset.state_ids[state]),
        "scene": scene,
        "original_index": original,
        "case_id": int(dataset.impairment_cases[state]),
        "case": ("ideal", "gain_phase", "coupling", "position", "compound")[int(dataset.impairment_cases[state])],
        "snr": float(dataset.snrs[state, scene]),
        "K_true": true_count,
        "K_predicted": predicted_count,
        "count_correct": int(correct),
        "route": "not_applicable",
        "ospa": ospa(estimate, truth),
        "scene_rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else float("nan"),
        "matched_squared_error_sum": float(np.sum(errors ** 2)) if len(errors) else 0.0,
        "matched_source_count": int(len(errors)),
        "maximum_matched_error": maximum,
        "e2e_1deg": int(correct and maximum <= 1.0),
        "e2e_2deg": int(correct and maximum <= 2.0),
        "e2e_3deg": int(correct and maximum <= 3.0),
        "manifold_error": float("nan"),
        "manifold_coherence": float("nan"),
        "protocol_fingerprint": protocol_fingerprint(),
    }


@torch.no_grad()
def evaluate_setnet_pair(
    mat_path: Path,
    checkpoint_directory: Path,
    support_count: int,
    device: str,
    completed_rows: list[dict] | None = None,
) -> list[dict]:
    """Evaluate Cov/IQ-SetNet with 8 context or 8+4 support scenes."""
    if support_count not in (8, 12):
        raise ValueError("support_count must be 8 or 12")
    dataset = SessionDataset.load_external(mat_path, validate_identity=True)
    models = {}
    checkpoints = {}
    for representation, filename in (("cov", "covariance.pt"), ("iq", "iq.pt")):
        model, checkpoint = load_setnet_checkpoint(checkpoint_directory / filename, representation, device)
        if int(checkpoint["context_scenes"]) != support_count:
            raise ValueError(f"{filename} was trained with {checkpoint['context_scenes']} supports")
        models[representation], checkpoints[representation] = model, checkpoint

    rows = list(completed_rows or [])
    expected_per_state = 2 * len(dataset.target_scenes[0])
    completed = {
        state for state in range(len(dataset.state_ids))
        if sum(int(row["state_index"]) == state for row in rows) == expected_per_state
    }
    rows = [row for row in rows if int(row["state_index"]) in completed]
    with h5py.File(mat_path, "r") as handle:
        for state in range(len(dataset.state_ids)):
            if state in completed:
                continue
            snapshots = dataset.read_session(state)
            prepared = [_scene_representation(value) for value in snapshots]
            support = list(map(int, dataset.context_scenes[state]))
            if support_count == 12:
                support.extend(map(int, dataset.guard_scenes[state]))
            for representation, model in models.items():
                checkpoint = checkpoints[representation]
                for scene_value in dataset.target_scenes[state]:
                    scene = int(scene_value)
                    selected = support + [scene]
                    if representation == "cov":
                        signal = np.stack([prepared[index][0] for index in selected])
                    else:
                        signal = np.stack([prepared[index][1] for index in selected])
                        rms = np.sqrt(np.mean(signal * signal, axis=(1, 2, 3), keepdims=True) + 1e-8)
                        signal = signal / rms
                    power = np.asarray([prepared[index][2] for index in selected], np.float32)
                    power = (power - float(checkpoint["power_mean"])) / max(float(checkpoint["power_std"]), 1e-8)
                    with torch.amp.autocast("cuda", enabled=str(device).startswith("cuda")):
                        output = model(
                            torch.from_numpy(signal[None]).to(device),
                            torch.from_numpy(power[None, :, None]).to(device),
                        )
                    count = int(torch.argmax(output["count_logits"][0]).item()) + 1
                    confidence = torch.sigmoid(output["exist_logits"][0])
                    queries = torch.topk(confidence, count).indices
                    estimate = output["angles"][0, queries].cpu().numpy()
                    method = f"{representation}_setnet_support{support_count}"
                    rows.append(_metric_row(dataset, handle, state, scene, estimate, method))
            if (state + 1) % 10 == 0 or state + 1 == len(dataset.state_ids):
                print(f"controlled SetNets support={support_count}: {state + 1}/{len(dataset.state_ids)} sessions")
    rows.sort(key=lambda row: (str(row["method"]), int(row["state_index"]), int(row["scene"])))
    return rows
