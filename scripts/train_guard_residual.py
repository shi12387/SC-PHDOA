"""Train the guard-conditioned 50-D residual calibration model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.calibration import SessionCalibrationNetwork
from sc_phdoa.calibration_training import stratified_state_split
from sc_phdoa.guard_features import build_guard_features, projection_score, source_residual_token
from sc_phdoa.manifold import angles_from_mixing, decode_calibration, make_manifold, nominal_manifold
from sc_phdoa.protocol import load_protocol, output_path

PROTOCOL = load_protocol()
SESSION_CACHE = output_path("session_calibration_cache")
CALIBRATION_CHECKPOINT = ROOT / "checkpoints" / "session_calibration" / "network.pt"
OUTPUT_DIRECTORY = output_path("guard_residual_training")
SEED = int(PROTOCOL["random_seed"])
RIDGE_ALPHA = 3000.0
CONTEXT_SCENES = int(PROTOCOL["context_scenes"])
GUARD_SCENES = int(PROTOCOL["guard_scenes"])
TARGET_SCENES = int(PROTOCOL["target_scenes"])


def scene_partitions(state_count: int, scene_count: int) -> tuple[np.ndarray, ...]:
    expected = CONTEXT_SCENES + GUARD_SCENES + TARGET_SCENES
    if scene_count != expected:
        raise ValueError(f"expected {expected} scenes per state, found {scene_count}")
    context = np.zeros((state_count, CONTEXT_SCENES), np.int64)
    guard = np.zeros((state_count, GUARD_SCENES), np.int64)
    target = np.zeros((state_count, TARGET_SCENES), np.int64)
    for state in range(state_count):
        order = np.random.default_rng(SEED + 104729 * state).permutation(scene_count)
        context_end = CONTEXT_SCENES
        guard_end = context_end + GUARD_SCENES
        context[state] = np.sort(order[:context_end])
        guard[state] = np.sort(order[context_end:guard_end])
        target[state] = np.sort(order[guard_end:])
    return context, guard, target


@torch.inference_mode()
def predict_calibration(
    model: SessionCalibrationNetwork,
    arrays: dict[str, np.ndarray],
    context_scenes: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    state_count = len(arrays["state_ids"])
    calibration = np.zeros((state_count, 50), np.float32)
    gates = np.zeros((state_count, 3), np.float32)
    for state in range(state_count):
        selected = context_scenes[state]
        output = model(
            torch.from_numpy(arrays["features"][state, selected][None]).to(device),
            torch.from_numpy(arrays["source_mask"][state, selected][None]).to(device),
            torch.from_numpy(np.asarray(
                arrays["aligned_iq"][state, selected][None], np.float32
            )).to(device),
        )
        calibration[state] = output["final"].cpu().numpy()[0]
        gates[state] = output["gates"].cpu().numpy()[0]
    return calibration, gates


def build_residual_features(
    arrays: dict[str, np.ndarray],
    initial: np.ndarray,
    gates: np.ndarray,
    guard_scenes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nominal_coordinates = np.asarray(arrays["nominal"], np.float64)
    spacing = float(arrays["spacing"])
    wavelength = float(arrays["wavelength"])
    pairs = [tuple(map(int, pair)) for pair in arrays["pairs"]]
    nominal = nominal_manifold(nominal_coordinates, wavelength)
    summaries = np.zeros((len(initial), 7), np.float32)
    features = np.zeros((len(initial), 190), np.float32)
    for state in range(len(initial)):
        calibration = decode_calibration(initial[state], nominal_coordinates, spacing, pairs)
        context = make_manifold(calibration, wavelength)
        calibration_norm = float(np.linalg.norm(initial[state]) / np.sqrt(50.0))
        nominal_scores, context_scores, tokens = [], [], []
        for scene in guard_scenes[state]:
            source_count = int(arrays["counts"][state, scene])
            mixing = (
                arrays["mixing_real"][state, scene, :source_count]
                + 1j * arrays["mixing_imag"][state, scene, :source_count]
            ).T.astype(np.complex128)
            nominal_scores.append(projection_score(mixing, nominal))
            context_scores.append(projection_score(mixing, context))
            angles = angles_from_mixing(mixing, context)
            predicted = context(angles)
            for source in range(source_count):
                tokens.append(source_residual_token(
                    mixing[:, source], predicted[:, source], angles[source], source_count,
                    float(arrays["snrs"][state, scene]), gates[state], calibration_norm,
                ))
        summary, full = build_guard_features(
            initial[state], gates[state], float(np.mean(nominal_scores)),
            float(np.mean(context_scores)), tokens,
        )
        summaries[state], features[state] = summary, full
        if (state + 1) % 25 == 0 or state + 1 == len(initial):
            print(f"guard features {state + 1}/{len(initial)}")
    return summaries, features


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train guard residual correction")
    parser.add_argument("--cache", type=Path, default=SESSION_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=CALIBRATION_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    arguments.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(np.load(arguments.cache / "metadata.npz", allow_pickle=False))
    metadata["aligned_iq"] = np.load(
        arguments.cache / "aligned_iq_float16.npy", mmap_mode="r"
    )
    device = torch.device(arguments.device)
    checkpoint = torch.load(
        arguments.checkpoint, map_location=device, weights_only=True
    )
    calibrator = SessionCalibrationNetwork().to(device)
    calibrator.load_state_dict(checkpoint["model"], strict=True)
    calibrator.eval()
    context_scenes, guard_scenes, target_scenes = scene_partitions(
        len(metadata["state_ids"]), metadata["features"].shape[1]
    )
    initial, gates = predict_calibration(calibrator, metadata, context_scenes, device)
    guard_summary, residual_features = build_residual_features(
        metadata, initial, gates, guard_scenes
    )
    target_residual = metadata["targets"] - initial
    splits = stratified_state_split(metadata["state_cases"], SEED)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])
    model.fit(residual_features[splits["train"]], target_residual[splits["train"]])
    prediction = np.asarray(model.predict(residual_features), np.float64)
    corrected = np.clip(initial + prediction, -1.0, 1.0)
    validation_rmse = float(np.sqrt(np.mean(
        (corrected[splits["validation"]] - metadata["targets"][splits["validation"]]) ** 2
    )))
    joblib.dump(model, arguments.output / "residual_model.joblib")
    np.savez_compressed(
        arguments.output / "guard_training_data.npz",
        initial=initial,
        gates=gates,
        targets=metadata["targets"],
        state_cases=metadata["state_cases"],
        guard_features=guard_summary,
        residual_features=residual_features,
        corrected=corrected.astype(np.float32),
        context_scenes=context_scenes,
        guard_scenes=guard_scenes,
        target_scenes=target_scenes,
        **splits,
    )
    report = {
        "seed": SEED,
        "ridge_alpha": RIDGE_ALPHA,
        "corrected_dimensions": 50,
        "feature_dimension": 190,
        "validation_parameter_rmse": validation_rmse,
        "train_states": len(splits["train"]),
        "validation_states": len(splits["validation"]),
        "test_states": len(splits["test"]),
    }
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Saved: {arguments.output}")


if __name__ == "__main__":
    main()
