"""Evaluate the final pipeline on the untouched external test set."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.evaluation import (
    evaluate_pipeline,
    load_completed_evaluation,
    paired_session_bootstrap,
    save_standard_reports,
    summarize,
    write_csv,
)
from sc_phdoa.inference import SCPHDOAInferencePipeline
from sc_phdoa.protocol import dataset_path, output_path

MAT_PATH = dataset_path("external_test")
OUTPUT_DIRECTORY = output_path("overall_comparison")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    if not MAT_PATH.is_file():
        raise FileNotFoundError(f"Missing external test set: {MAT_PATH}")
    pipeline = SCPHDOAInferencePipeline.from_files(
        MAT_PATH, device=DEVICE, validate_identity=True
    )
    expected_states = pipeline.state_count
    expected_rows = sum(
        len(scenes) for scenes in pipeline.backend.dataset.target_scenes
    )
    combined = []
    for method, route in (
        ("nominal_als", "nominal"),
        ("session_calibrated_als", "session"),
        ("sc_phdoa", "guard"),
    ):
        method_output = OUTPUT_DIRECTORY / method
        rows = load_completed_evaluation(
            method_output / "scene_details.csv", expected_rows, expected_states
        )
        if rows is None:
            print(f"evaluating {method}")
            rows = evaluate_pipeline(
                pipeline,
                MAT_PATH,
                method_output,
                calibration_route=route,
                reuse_preprocessing=True,
            )
        for row in rows:
            row["method"] = method
        save_standard_reports(rows, method_output)
        combined.extend(rows)
        write_csv(OUTPUT_DIRECTORY / "scene_details.csv", combined)
        write_csv(
            OUTPUT_DIRECTORY / "method_summary.csv",
            summarize(combined, ("method",)),
        )
    comparisons = []
    for method_a, method_b in (
        ("nominal_als", "session_calibrated_als"),
        ("nominal_als", "sc_phdoa"),
        ("session_calibrated_als", "sc_phdoa"),
    ):
        comparisons.extend(paired_session_bootstrap(
            combined, method_a, method_b
        ))
    write_csv(OUTPUT_DIRECTORY / "paired_session_bootstrap.csv", comparisons)


if __name__ == "__main__":
    main()
