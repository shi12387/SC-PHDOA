"""Evaluate ID, mild-OOD, and strong-OOD impairment distributions."""

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
    summarize,
    write_csv,
)
from sc_phdoa.inference import SCPHDOAInferencePipeline
from sc_phdoa.protocol import dataset_path, output_path

DATASETS = {
    "id": dataset_path("external_test"),
    "mild_ood": dataset_path("mild_ood"),
    "strong_ood": dataset_path("strong_ood"),
}
OUTPUT_DIRECTORY = output_path("ood_generalization")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    missing = [str(path) for path in DATASETS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing OOD evaluation datasets:\n  " + "\n  ".join(missing)
        )
    combined = []
    for tier, mat_path in DATASETS.items():
        print(f"\n===== distribution: {tier} =====")
        pipeline = SCPHDOAInferencePipeline.from_files(
            mat_path, device=DEVICE, validate_identity=tier == "id"
        )
        expected_states = pipeline.state_count
        expected_rows = sum(
            len(scenes) for scenes in pipeline.backend.dataset.target_scenes
        )
        for method, route in (
            ("nominal_als", "nominal"),
            ("session_calibrated_als", "session"),
            ("sc_phdoa", "guard"),
        ):
            method_output = OUTPUT_DIRECTORY / tier / method
            rows = load_completed_evaluation(
                method_output / "scene_details.csv", expected_rows, expected_states
            )
            if rows is None and tier == "id":
                rows = load_completed_evaluation(
                    output_path("overall_comparison") / method / "scene_details.csv",
                    expected_rows,
                    expected_states,
                )
                if rows is not None:
                    write_csv(method_output / "scene_details.csv", rows)
            if rows is None:
                print(f"evaluating {tier}/{method}")
                rows = evaluate_pipeline(
                    pipeline,
                    mat_path,
                    method_output,
                    calibration_route=route,
                    reuse_preprocessing=True,
                )
            for row in rows:
                row["distribution"] = tier
                row["method"] = method
            combined.extend(rows)
            write_csv(OUTPUT_DIRECTORY / "scene_details.csv", combined)
            write_csv(
                OUTPUT_DIRECTORY / "distribution_summary.csv",
                summarize(combined, ("distribution", "method")),
            )
    comparisons = []
    relative = []
    for tier in DATASETS:
        tier_rows = [row for row in combined if row["distribution"] == tier]
        for row in paired_session_bootstrap(
            tier_rows, "nominal_als", "sc_phdoa"
        ):
            row["distribution"] = tier
            comparisons.append(row)
        summary = {
            row["method"]: row
            for row in summarize(tier_rows, ("method",))
        }
        nominal_error = float(summary["nominal_als"]["manifold_error"])
        sc_error = float(summary["sc_phdoa"]["manifold_error"])
        relative.append({
            "distribution": tier,
            "nominal_manifold_error": nominal_error,
            "sc_phdoa_manifold_error": sc_error,
            "relative_reduction": (
                (nominal_error - sc_error) / nominal_error
                if nominal_error > 0.0 else float("nan")
            ),
        })
    write_csv(OUTPUT_DIRECTORY / "paired_session_bootstrap.csv", comparisons)
    write_csv(OUTPUT_DIRECTORY / "manifold_relative_reduction.csv", relative)


if __name__ == "__main__":
    main()
