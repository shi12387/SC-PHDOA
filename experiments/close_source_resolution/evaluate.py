"""Evaluate spatial-only and joint-hard close-source resolution."""

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
    summarize,
    write_csv,
)
from sc_phdoa.inference import SCPHDOAInferencePipeline
from sc_phdoa.protocol import dataset_path, output_path

DATASETS = {
    "spatial_only": dataset_path("close_source_spatial_only"),
    "joint_hard": dataset_path("close_source_joint_hard"),
}
OUTPUT_DIRECTORY = output_path("close_source_resolution")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def matlab_vector(handle: h5py.File, name: str) -> np.ndarray:
    return np.asarray(handle[name]).reshape(-1)


def main() -> None:
    missing = [str(path) for path in DATASETS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing close-source datasets:\n  " + "\n  ".join(missing)
        )
    combined = []
    for mode, mat_path in DATASETS.items():
        with h5py.File(mat_path, "r") as handle:
            target_separation = matlab_vector(handle, "target_separation_deg_all")
            actual_separation = matlab_vector(handle, "min_sep_all")
            delay_separation = matlab_vector(handle, "min_delay_separation_all")
            cfo_separation = matlab_vector(handle, "min_cfo_separation_all")
        pipeline = SCPHDOAInferencePipeline.from_files(
            mat_path, device=DEVICE, validate_identity=False
        )
        expected_states = pipeline.state_count
        expected_rows = sum(
            len(scenes) for scenes in pipeline.backend.dataset.target_scenes
        )
        for method, route in (("nominal_als", "nominal"), ("sc_phdoa", "guard")):
            method_output = OUTPUT_DIRECTORY / mode / method
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
                )
            for row in rows:
                original = int(row["original_index"])
                requested = float(target_separation[original])
                threshold = min(3.0, requested / 2.0)
                row.update({
                    "mode": mode,
                    "method": method,
                    "target_separation_deg": requested,
                    "actual_min_separation_deg": float(actual_separation[original]),
                    "min_delay_separation_samples": float(delay_separation[original]),
                    "min_cfo_separation": float(cfo_separation[original]),
                    "resolution_threshold_deg": threshold,
                    "resolution_success": int(
                        row["count_correct"]
                        and row["maximum_matched_error"] <= threshold
                    ),
                })
            combined.extend(rows)
    write_csv(OUTPUT_DIRECTORY / "scene_details.csv", combined)
    summary = summarize(
        combined, ("mode", "method", "target_separation_deg", "K_true")
    )
    lookup = {
        (row["mode"], row["method"], row["target_separation_deg"], row["K_true"]): row
        for row in summary
    }
    for key, row in lookup.items():
        selected = [
            item for item in combined
            if (
                item["mode"], item["method"],
                item["target_separation_deg"], item["K_true"]
            ) == key
        ]
        row["resolution_success"] = float(np.mean([item["resolution_success"] for item in selected]))
    write_csv(OUTPUT_DIRECTORY / "resolution_summary.csv", summary)


if __name__ == "__main__":
    main()
