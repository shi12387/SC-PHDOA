"""Evaluate matched versus same-condition shuffled session calibration."""

from __future__ import annotations

import sys
from pathlib import Path

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
from sc_phdoa.protocol import dataset_path, load_protocol, output_path

PROTOCOL = load_protocol()
MAT_PATH = dataset_path("external_test")
OUTPUT_DIRECTORY = output_path("shuffled_session")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(PROTOCOL["random_seed"])


def same_condition_derangement(cases: np.ndarray) -> dict[int, int]:
    """Return a deterministic donor map with no self mappings."""
    rng = np.random.default_rng(SEED + 12000)
    donors: dict[int, int] = {}
    for case in sorted(np.unique(cases)):
        states = np.flatnonzero(cases == case).astype(np.int64)
        if len(states) < 2:
            raise ValueError(f"condition {case} has fewer than two sessions")
        order = rng.permutation(states)
        shifted = np.roll(order, 1)
        donors.update({int(target): int(donor) for target, donor in zip(order, shifted)})
    if any(state == donor for state, donor in donors.items()):
        raise RuntimeError("same-condition shuffle contains a self mapping")
    return donors


def main() -> None:
    pipeline = SCPHDOAInferencePipeline.from_files(
        MAT_PATH, device=DEVICE, validate_identity=True
    )
    dataset = pipeline.backend.dataset
    expected_states = pipeline.state_count
    expected_rows = sum(len(scenes) for scenes in dataset.target_scenes)
    donors = same_condition_derangement(dataset.impairment_cases)
    write_csv(
        OUTPUT_DIRECTORY / "donor_mapping.csv",
        [
            {
                "state_index": state,
                "donor_state_index": donor,
                "case_id": int(dataset.impairment_cases[state]),
            }
            for state, donor in sorted(donors.items())
        ],
    )

    combined = []
    for method, donor_map in (("matched", None), ("same_condition_shuffled", donors)):
        directory = OUTPUT_DIRECTORY / method
        rows = load_completed_evaluation(
            directory / "scene_details.csv", expected_rows, expected_states
        )
        if rows is None and method == "matched":
            rows = load_completed_evaluation(
                output_path("overall_comparison") / "session_calibrated_als" / "scene_details.csv",
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
                calibration_route="session",
                reuse_preprocessing=True,
                calibration_source_by_state=donor_map,
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
        OUTPUT_DIRECTORY / "condition_summary.csv",
        summarize(combined, ("method", "case")),
    )
    comparisons = paired_session_bootstrap(
        combined, "matched", "same_condition_shuffled"
    )
    for case in sorted(set(row["case"] for row in combined)):
        selected = [row for row in combined if row["case"] == case]
        for comparison in paired_session_bootstrap(
            selected, "matched", "same_condition_shuffled"
        ):
            comparison["case"] = case
            comparisons.append(comparison)
    write_csv(OUTPUT_DIRECTORY / "paired_session_bootstrap.csv", comparisons)


if __name__ == "__main__":
    main()
