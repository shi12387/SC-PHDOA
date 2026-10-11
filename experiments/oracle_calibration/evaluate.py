"""Compare nominal, learned, guard-routed, and oracle-calibrated manifolds."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.dataset import read_calibration, read_doas
from sc_phdoa.evaluation import (
    CASE_NAMES,
    evaluate_pipeline,
    load_completed_evaluation,
    paired_session_bootstrap,
    read_csv,
    summarize,
    write_csv,
)
from sc_phdoa.inference import SCPHDOAInferencePipeline
from sc_phdoa.manifold import COARSE_GRID, make_manifold
from sc_phdoa.metrics import matched_errors, ospa
from sc_phdoa.protocol import dataset_path, output_path
from sc_phdoa.protocol import protocol_fingerprint

MAT_PATH = dataset_path("external_test")
OUTPUT_DIRECTORY = output_path("oracle_calibration")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def oracle_rows(pipeline: SCPHDOAInferencePipeline) -> list[dict]:
    backend = pipeline.backend
    dataset = backend.dataset
    output_directory = OUTPUT_DIRECTORY / "oracle_calibrated_als"
    output_directory.mkdir(parents=True, exist_ok=True)
    scene_path = output_directory / "scene_details.csv"
    expected_states = pipeline.state_count
    expected_rows = sum(len(scenes) for scenes in dataset.target_scenes)
    complete = load_completed_evaluation(
        scene_path, expected_rows, expected_states
    )
    if complete is not None:
        return complete

    rows = read_csv(scene_path) if scene_path.is_file() else []
    fingerprint = protocol_fingerprint()
    if rows and any(
        row.get("protocol_fingerprint") != fingerprint
        or "manifold_error" not in row
        for row in rows
    ):
        print(f"discarding incompatible oracle cache: {scene_path}")
        rows = []
    completed_states = {int(row["state_index"]) for row in rows}
    if completed_states:
        print(
            f"resuming oracle evaluation: {len(completed_states)}/"
            f"{pipeline.state_count} sessions complete"
        )
    with h5py.File(MAT_PATH, "r") as handle:
        for state in range(pipeline.state_count):
            if state in completed_states:
                continue
            snapshots_by_scene = backend.read_session(state)
            frontends = {}
            for scene_value in dataset.target_scenes[state]:
                scene = int(scene_value)
                snapshots = snapshots_by_scene[scene]
                count_prediction = pipeline.model.predict_source_count(
                    snapshots, backend.payload_start, backend.preamble, backend.maximum_delay
                )
                snr = backend.estimate_snr(snapshots, count_prediction.count)
                frontends[scene] = backend.build_source_frontend(
                    snapshots, count_prediction.count, snr
                )
            calibration_index = int(dataset.original_indices[state, 0])
            calibration = read_calibration(handle, calibration_index, dataset.sample_count)
            manifold = make_manifold(calibration, dataset.wavelength)
            routed = {"manifold": manifold, "grid": manifold(COARSE_GRID)}
            for scene in dataset.target_scenes[state]:
                scene = int(scene)
                original = int(dataset.original_indices[state, scene])
                true_count = int(dataset.source_counts[state, scene])
                predicted_count = int(frontends[scene]["count"])
                angles, _ = backend.estimate_target_doas(
                    snapshots_by_scene[scene], frontends[scene], routed
                )
                truth = read_doas(handle, original, dataset.sample_count, true_count)
                count_correct = predicted_count == true_count
                errors = matched_errors(angles, truth) if count_correct else np.asarray([])
                maximum = float(np.max(errors)) if len(errors) else float("nan")
                rows.append({
                    "method": "oracle_calibrated_als",
                    "state_index": state,
                    "state_id": int(dataset.state_ids[state]),
                    "scene": scene,
                    "original_index": original,
                    "case_id": int(dataset.impairment_cases[state]),
                    "case": CASE_NAMES.get(
                        int(dataset.impairment_cases[state]),
                        str(int(dataset.impairment_cases[state])),
                    ),
                    "snr": float(dataset.snrs[state, scene]),
                    "K_true": true_count,
                    "K_predicted": predicted_count,
                    "count_correct": int(count_correct),
                    "route": "oracle",
                    "ospa": ospa(angles, truth),
                    "scene_rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else float("nan"),
                    "matched_squared_error_sum": float(np.sum(errors**2)) if len(errors) else 0.0,
                    "matched_source_count": int(len(errors)),
                    "maximum_matched_error": maximum,
                    "e2e_1deg": int(count_correct and maximum <= 1.0),
                    "e2e_2deg": int(count_correct and maximum <= 2.0),
                    "e2e_3deg": int(count_correct and maximum <= 3.0),
                    "manifold_error": 0.0,
                    "manifold_coherence": 1.0,
                    "protocol_fingerprint": protocol_fingerprint(),
                })
            if (state + 1) % 10 == 0 or state + 1 == pipeline.state_count:
                print(f"oracle sessions {state + 1}/{pipeline.state_count}")
            write_csv(scene_path, rows)
    return rows


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
        ("guard_routed_als", "guard"),
    ):
        method_output = OUTPUT_DIRECTORY / method
        rows = load_completed_evaluation(
            method_output / "scene_details.csv", expected_rows, expected_states
        )
        shared_name = "sc_phdoa" if method == "guard_routed_als" else method
        if rows is None:
            rows = load_completed_evaluation(
                output_path("overall_comparison") / shared_name / "scene_details.csv",
                expected_rows,
                expected_states,
            )
            if rows is not None:
                write_csv(method_output / "scene_details.csv", rows)
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

    oracle = oracle_rows(pipeline)
    for row in oracle:
        row["method"] = "oracle_calibrated_als"
    combined.extend(oracle)
    write_csv(OUTPUT_DIRECTORY / "scene_details.csv", combined)
    write_csv(
        OUTPUT_DIRECTORY / "method_summary.csv", summarize(combined, ("method",))
    )
    write_csv(
        OUTPUT_DIRECTORY / "condition_summary.csv",
        summarize(combined, ("method", "case")),
    )
    comparisons = []
    for method_a, method_b in (
        ("nominal_als", "session_calibrated_als"),
        ("nominal_als", "guard_routed_als"),
        ("nominal_als", "oracle_calibrated_als"),
        ("session_calibrated_als", "guard_routed_als"),
    ):
        comparisons.extend(paired_session_bootstrap(
            combined, method_a, method_b
        ))
    write_csv(OUTPUT_DIRECTORY / "paired_session_bootstrap.csv", comparisons)


if __name__ == "__main__":
    main()
