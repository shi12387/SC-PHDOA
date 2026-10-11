"""Compare deployable predicted-K and target-only Oracle-K SC-PHDOA."""

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

MAT_PATH = dataset_path("external_test")
OUTPUT_DIRECTORY = output_path("oracle_source_count")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    pipeline = SCPHDOAInferencePipeline.from_files(
        MAT_PATH, device=DEVICE, validate_identity=True
    )
    dataset = pipeline.backend.dataset
    expected_states = pipeline.state_count
    expected_rows = sum(len(scenes) for scenes in dataset.target_scenes)
    target_counts = {
        state: {
            int(scene): int(dataset.source_counts[state, int(scene)])
            for scene in dataset.target_scenes[state]
        }
        for state in range(expected_states)
    }
    combined = []
    for method, overrides in (
        ("predicted_k", None),
        ("oracle_target_k", target_counts),
    ):
        directory = OUTPUT_DIRECTORY / method
        rows = load_completed_evaluation(directory / "scene_details.csv", expected_rows, expected_states)
        if rows is None and method == "predicted_k":
            rows = load_completed_evaluation(
                output_path("overall_comparison") / "sc_phdoa" / "scene_details.csv",
                expected_rows,
                expected_states,
            )
            if rows is not None:
                write_csv(directory / "scene_details.csv", rows)
        if rows is None:
            rows = evaluate_pipeline(
                pipeline,
                MAT_PATH,
                directory,
                calibration_route="guard",
                reuse_preprocessing=True,
                target_counts_by_state=overrides,
            )
        for row in rows:
            row["method"] = method
        combined.extend(rows)
    write_csv(OUTPUT_DIRECTORY / "scene_details.csv", combined)
    write_csv(
        OUTPUT_DIRECTORY / "method_summary.csv",
        summarize(combined, ("method",)),
    )
    write_csv(
        OUTPUT_DIRECTORY / "source_count_summary.csv",
        summarize(combined, ("method", "K_true")),
    )
    write_csv(
        OUTPUT_DIRECTORY / "paired_session_bootstrap.csv",
        paired_session_bootstrap(combined, "predicted_k", "oracle_target_k"),
    )


if __name__ == "__main__":
    main()
