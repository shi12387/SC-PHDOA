"""Select the session-level guard routing threshold on validation states."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.als import calibrated_als
from sc_phdoa.config import DEFAULT_CONFIG
from sc_phdoa.dataset import read_doas, read_iq
from sc_phdoa.manifold import COARSE_GRID, decode_calibration, make_manifold, nominal_manifold
from sc_phdoa.metrics import matched_errors
from sc_phdoa.protocol import dataset_path, output_path

TRAINING_MAT = dataset_path("session_calibration_training")
SESSION_CACHE = output_path("session_calibration_cache")
GUARD_CACHE = output_path("guard_residual_training") / "guard_training_data.npz"
OUTPUT_DIRECTORY = output_path("guard_threshold_selection")

# The released threshold was selected before the final ALS-depth change and
# was subsequently frozen. Keeping this value reproduces that provenance.
SELECTION_ALS_ITERATIONS = 5


def evaluate_branches(
    mat_path: Path,
    arrays: dict[str, np.ndarray],
    guard: dict[str, np.ndarray],
) -> list[dict[str, float]]:
    nominal_coordinates = np.asarray(arrays["nominal"], np.float64)
    spacing = float(arrays["spacing"])
    wavelength = float(arrays["wavelength"])
    pairs = [tuple(map(int, pair)) for pair in arrays["pairs"]]
    nominal = nominal_manifold(nominal_coordinates, wavelength)
    nominal_grid = nominal(COARSE_GRID)
    config = replace(DEFAULT_CONFIG, als_iterations=SELECTION_ALS_ITERATIONS)
    rows = []
    validation_states = np.asarray(guard["validation"], np.int64)
    with h5py.File(mat_path, "r") as handle:
        sample_count = int(arrays["originals"].size)
        for position, state in enumerate(validation_states, 1):
            candidate_calibration = decode_calibration(
                guard["corrected"][state], nominal_coordinates, spacing, pairs
            )
            candidate = make_manifold(candidate_calibration, wavelength)
            candidate_grid = candidate(COARSE_GRID)
            for scene in guard["target_scenes"][state]:
                source_count = int(arrays["counts"][state, scene])
                original = int(arrays["originals"][state, scene])
                snapshots = read_iq(handle, original, sample_count)
                target = read_doas(handle, original, sample_count, source_count)
                mixing = (
                    arrays["mixing_real"][state, scene, :source_count]
                    + 1j * arrays["mixing_imag"][state, scene, :source_count]
                ).T.astype(np.complex128)
                branch_metrics = {}
                for name, manifold, grid in (
                    ("nominal", nominal, nominal_grid),
                    ("candidate", candidate, candidate_grid),
                ):
                    _, angles, _ = calibrated_als(
                        snapshots, mixing, manifold, grid, config
                    )
                    errors = matched_errors(angles, target)
                    branch_metrics[f"{name}_rmse"] = float(np.sqrt(np.mean(errors**2)))
                    branch_metrics[f"{name}_maximum"] = float(np.max(errors))
                rows.append({
                    "state": int(state),
                    "guard_score": float(guard["guard_features"][state, 2]),
                    **branch_metrics,
                })
            if position % 10 == 0 or position == len(validation_states):
                print(f"validation states {position}/{len(validation_states)}")
    return rows


def threshold_metrics(rows: list[dict[str, float]], threshold: float) -> dict[str, float]:
    rmse, maximum = [], []
    routed_states = set()
    states = set()
    for row in rows:
        use_candidate = row["guard_score"] > threshold
        states.add(int(row["state"]))
        if use_candidate:
            routed_states.add(int(row["state"]))
        prefix = "candidate" if use_candidate else "nominal"
        rmse.append(row[f"{prefix}_rmse"])
        maximum.append(row[f"{prefix}_maximum"])
    rmse_mean = float(np.mean(rmse))
    accuracy_two = float(np.mean(np.asarray(maximum) <= 2.0))
    accuracy_three = float(np.mean(np.asarray(maximum) <= 3.0))
    objective = rmse_mean + 0.60 * (1.0 - accuracy_three) + 0.20 * (1.0 - accuracy_two)
    return {
        "threshold": float(threshold),
        "objective": objective,
        "rmse_mean": rmse_mean,
        "angular_success_2deg": accuracy_two,
        "angular_success_3deg": accuracy_three,
        "activation_rate": len(routed_states) / max(len(states), 1),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select guard routing threshold")
    parser.add_argument("--mat", type=Path, default=TRAINING_MAT)
    parser.add_argument("--session-cache", type=Path, default=SESSION_CACHE)
    parser.add_argument("--guard-cache", type=Path, default=GUARD_CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    arguments.output.mkdir(parents=True, exist_ok=True)
    arrays = dict(np.load(arguments.session_cache / "metadata.npz", allow_pickle=False))
    guard = dict(np.load(arguments.guard_cache, allow_pickle=False))
    rows = evaluate_branches(arguments.mat, arrays, guard)
    validation_states = np.asarray(guard["validation"], np.int64)
    scores = guard["guard_features"][validation_states, 2]
    thresholds = np.unique(np.r_[
        -np.inf,
        np.quantile(scores, np.linspace(0.0, 1.0, 31)),
        np.inf,
    ])
    scan = [threshold_metrics(rows, float(threshold)) for threshold in thresholds]
    best = min(scan, key=lambda row: row["objective"])
    with (arguments.output / "threshold_scan.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=scan[0].keys())
        writer.writeheader()
        writer.writerows(scan)
    report = {
        "selected": best,
        "selection_als_iterations": SELECTION_ALS_ITERATIONS,
        "candidate_branch": "residual_corrected_calibrated_manifold",
        "non_candidate_branch": "nominal_manifold",
        "candidate_thresholds": len(thresholds),
    }
    (arguments.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (arguments.output / "threshold.json").write_text(
        json.dumps(
            {
                "threshold": best["threshold"],
                "score": "nominal_projection_error_minus_context_projection_error",
                "residual_route_when": "score > threshold",
                "selection_als_iterations": SELECTION_ALS_ITERATIONS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
