"""Measure session-calibration sensitivity to the number of context scenes."""

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
    summarize,
    write_csv,
)
from sc_phdoa.inference import SCPHDOAInferencePipeline
from sc_phdoa.protocol import dataset_path, load_protocol, output_path

PROTOCOL = load_protocol()
MAT_PATH = dataset_path("external_test")
OUTPUT_DIRECTORY = output_path("context_sensitivity")
CONTEXT_COUNTS = tuple(PROTOCOL["context_counts"])
REPEATS = int(PROTOCOL["context_subsampling_repeats"])
SEED = int(PROTOCOL["random_seed"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INTEGER_FIELDS = {
    "state_index", "state_id", "scene", "original_index", "case_id",
    "K_true", "K_predicted", "count_correct", "matched_source_count",
    "e2e_1deg", "e2e_2deg", "e2e_3deg",
}
FLOAT_FIELDS = {
    "snr", "ospa", "scene_rmse", "matched_squared_error_sum",
    "maximum_matched_error",
}


def context_selection(
    available: np.ndarray, context_count: int, repeat: int
) -> dict[int, np.ndarray]:
    output = {}
    for state in range(len(available)):
        if context_count == available.shape[1]:
            output[state] = available[state]
        else:
            rng = np.random.default_rng(
                SEED + 18200 + state * 1009 + repeat * 37
            )
            permutation = rng.permutation(np.asarray(available[state], np.int64))
            output[state] = np.sort(permutation[:context_count])
    return output


def write_progress(rows: list[dict]) -> list[dict]:
    """Persist aggregate outputs after every repeat."""
    write_csv(OUTPUT_DIRECTORY / "scene_details.csv", rows)
    per_repeat = summarize(rows, ("context_count", "repeat"))
    write_csv(OUTPUT_DIRECTORY / "repeat_summary.csv", per_repeat)

    final = []
    for context_count in CONTEXT_COUNTS:
        selected = [
            row for row in per_repeat if row["context_count"] == context_count
        ]
        if not selected:
            continue
        final.append({
            "context_count": context_count,
            "repeats": len(selected),
            "mean_ospa": float(np.mean([row["mean_ospa"] for row in selected])),
            "std_ospa": float(np.std([row["mean_ospa"] for row in selected])),
            "mean_e2e_3deg": float(np.mean([row["e2e_3deg"] for row in selected])),
            "std_e2e_3deg": float(np.std([row["e2e_3deg"] for row in selected])),
            "mean_scene_correct_k_rmse": float(np.mean([
                row["mean_scene_correct_k_rmse"] for row in selected
            ])),
            "std_scene_correct_k_rmse": float(np.std([
                row["mean_scene_correct_k_rmse"] for row in selected
            ])),
            "mean_manifold_error": float(np.mean([
                row["manifold_error"] for row in selected
            ])),
            "std_manifold_error": float(np.std([
                row["manifold_error"] for row in selected
            ])),
            "mean_manifold_coherence": float(np.mean([
                row["manifold_coherence"] for row in selected
            ])),
            "std_manifold_coherence": float(np.std([
                row["manifold_coherence"] for row in selected
            ])),
        })
    write_csv(OUTPUT_DIRECTORY / "context_summary.csv", final)
    return per_repeat


def main() -> None:
    pipeline = SCPHDOAInferencePipeline.from_files(
        MAT_PATH, device=DEVICE, validate_identity=True
    )
    dataset = pipeline.backend.dataset
    expected_states = pipeline.state_count
    expected_rows = sum(len(scenes) for scenes in dataset.target_scenes)
    combined = []
    for context_count in CONTEXT_COUNTS:
        repeat_count = 1 if context_count == 8 else REPEATS
        for repeat in range(repeat_count):
            repeat_directory = (
                OUTPUT_DIRECTORY / f"context_{context_count}" / f"repeat_{repeat + 1}"
            )
            scene_path = repeat_directory / "scene_details.csv"
            rows = load_completed_evaluation(
                scene_path, expected_rows, expected_states
            )
            if rows is not None:
                print(
                    f"context={context_count}, repeat={repeat + 1}: "
                    f"reusing {scene_path}"
                )
            else:
                print(f"context={context_count}, repeat={repeat + 1}: evaluating")
                rows = evaluate_pipeline(
                    pipeline,
                    MAT_PATH,
                    repeat_directory,
                    calibration_route="session",
                    context_scenes_by_state=context_selection(
                        dataset.context_scenes, context_count, repeat
                    ),
                    reuse_preprocessing=True,
                )
            for row in rows:
                row["context_count"] = context_count
                row["repeat"] = repeat + 1
            combined.extend(rows)
            write_progress(combined)


if __name__ == "__main__":
    main()
