"""Evaluate controlled context-target calibration drift."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.evaluation import (
    evaluate_pipeline,
    load_completed_evaluation,
    paired_session_bootstrap,
    summarize,
    write_csv,
)
from sc_phdoa.inference import SCPHDOAInferencePipeline
from sc_phdoa.dataset import read_calibration
from sc_phdoa.manifold import COARSE_GRID, make_manifold
from sc_phdoa.protocol import dataset_path, output_path

DATASETS = {
    "gain_phase": dataset_path("drift_gain_phase"),
    "coupling": dataset_path("drift_coupling"),
    "position": dataset_path("drift_position"),
    "compound": dataset_path("drift_compound"),
}
OUTPUT_DIRECTORY = output_path("calibration_drift")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    missing = [str(path) for path in DATASETS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing calibration-drift datasets:\n  " + "\n  ".join(missing)
        )
    combined = []
    for component, mat_path in DATASETS.items():
        with h5py.File(mat_path, "r") as handle:
            drift = np.asarray(handle["drift_lambda_all"]).reshape(-1)
        pipeline = SCPHDOAInferencePipeline.from_files(
            mat_path, device=DEVICE, validate_identity=False
        )
        expected_states = pipeline.state_count
        expected_rows = sum(
            len(scenes) for scenes in pipeline.backend.dataset.target_scenes
        )
        dataset = pipeline.backend.dataset
        oracle_manifolds = {}
        with h5py.File(mat_path, "r") as handle:
            for state in range(expected_states):
                target_scene = int(dataset.target_scenes[state, 0])
                original = int(dataset.original_indices[state, target_scene])
                calibration = read_calibration(
                    handle, original, dataset.sample_count
                )
                physical = make_manifold(calibration, dataset.wavelength)
                oracle_manifolds[state] = {
                    "manifold": physical,
                    "grid": physical(COARSE_GRID),
                    "mode": "oracle",
                    "use_residual": False,
                    "guard_features": np.full(7, np.nan, np.float32),
                    "initial_calibration": np.zeros(50, np.float32),
                    "corrected_calibration": np.zeros(50, np.float32),
                }
        for method, route in (
            ("nominal_als", "nominal"),
            ("session_calibrated_als", "session"),
            ("guard_routed_als", "guard"),
            ("oracle_calibrated_als", "oracle"),
        ):
            method_output = OUTPUT_DIRECTORY / component / method
            rows = load_completed_evaluation(
                method_output / "scene_details.csv", expected_rows, expected_states
            )
            if rows is None:
                rows = evaluate_pipeline(
                    pipeline,
                    mat_path,
                    method_output,
                    calibration_route=route,
                    reuse_preprocessing=True,
                    manifold_by_state=(
                        oracle_manifolds if route == "oracle" else None
                    ),
                )
            for row in rows:
                row["drift_component"] = component
                row["drift_lambda"] = float(drift[int(row["original_index"])])
                row["method"] = method
            combined.extend(rows)
    write_csv(OUTPUT_DIRECTORY / "scene_details.csv", combined)
    write_csv(
        OUTPUT_DIRECTORY / "drift_summary.csv",
        summarize(combined, ("drift_component", "drift_lambda", "method")),
    )
    comparisons = []
    for component in DATASETS:
        levels = sorted({
            row["drift_lambda"] for row in combined
            if row["drift_component"] == component
        })
        for level in levels:
            selected = [
                row for row in combined
                if row["drift_component"] == component
                and row["drift_lambda"] == level
            ]
            for method_a, method_b in (
                ("nominal_als", "session_calibrated_als"),
                ("session_calibrated_als", "guard_routed_als"),
                ("nominal_als", "guard_routed_als"),
                ("guard_routed_als", "oracle_calibrated_als"),
            ):
                for result in paired_session_bootstrap(
                    selected, method_a, method_b
                ):
                    result["drift_component"] = component
                    result["drift_lambda"] = level
                    comparisons.append(result)
    write_csv(OUTPUT_DIRECTORY / "paired_session_bootstrap.csv", comparisons)


if __name__ == "__main__":
    main()
