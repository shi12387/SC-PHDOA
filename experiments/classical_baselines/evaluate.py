"""E01 MUSIC, MDL-MUSIC, and ESPRIT on the external ID set."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.baselines import NominalArray, l_shaped_esprit, music_2d, sample_covariance, source_count_mdl
from sc_phdoa.dataset import SessionDataset, read_doas
from sc_phdoa.evaluation import CASE_NAMES, read_csv, save_standard_reports, summarize, write_csv
from sc_phdoa.metrics import matched_errors, ospa
from sc_phdoa.protocol import dataset_path, output_path, protocol_fingerprint

MAT_PATH = dataset_path("external_test")
OUTPUT_DIRECTORY = output_path("classical_baselines")
COUNT_DETAILS = output_path("overall_comparison") / "nominal_als" / "scene_details.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def metric_row(dataset, handle, state, scene, estimate, method, fallback=0):
    original = int(dataset.original_indices[state, scene])
    true_count = int(dataset.source_counts[state, scene])
    truth = read_doas(handle, original, dataset.sample_count, true_count)
    predicted_count = len(estimate)
    correct = predicted_count == true_count
    errors = matched_errors(estimate, truth) if correct else np.asarray([])
    maximum = float(np.max(errors)) if len(errors) else float("nan")
    return {
        "method": method, "state_index": state,
        "state_id": int(dataset.state_ids[state]), "scene": scene,
        "original_index": original, "case_id": int(dataset.impairment_cases[state]),
        "case": CASE_NAMES[int(dataset.impairment_cases[state])],
        "snr": float(dataset.snrs[state, scene]), "K_true": true_count,
        "K_predicted": predicted_count, "count_correct": int(correct),
        "route": "not_applicable", "ospa": ospa(estimate, truth),
        "scene_rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else float("nan"),
        "matched_squared_error_sum": float(np.sum(errors ** 2)) if len(errors) else 0.0,
        "matched_source_count": int(len(errors)), "maximum_matched_error": maximum,
        "e2e_1deg": int(correct and maximum <= 1.0),
        "e2e_2deg": int(correct and maximum <= 2.0),
        "e2e_3deg": int(correct and maximum <= 3.0),
        "manifold_error": float("nan"), "manifold_coherence": float("nan"),
        "esprit_music_fallback": int(fallback),
        "protocol_fingerprint": protocol_fingerprint(),
    }


def main() -> None:
    if not COUNT_DETAILS.is_file():
        raise FileNotFoundError("Run E01 physical pipeline first; shared predicted-K rows are missing")
    count_rows = read_csv(COUNT_DETAILS)
    shared_counts = {(int(row["state_index"]), int(row["scene"])): int(row["K_predicted"]) for row in count_rows}
    dataset = SessionDataset.load_external(MAT_PATH, validate_identity=True)
    array = NominalArray(dataset.nominal_coordinates, dataset.wavelength, grid_step_deg=1.0)
    scene_path = OUTPUT_DIRECTORY / "scene_details.csv"
    rows = read_csv(scene_path) if scene_path.is_file() else []
    methods = {"mdl_music", "music", "esprit"}
    expected = len(methods) * len(dataset.target_scenes[0])
    rows = [row for row in rows if row["method"] in methods and row.get("protocol_fingerprint") == protocol_fingerprint()]
    completed = {
        state for state in range(len(dataset.state_ids))
        if len([row for row in rows if int(row["state_index"]) == state]) == expected
        and {row["method"] for row in rows if int(row["state_index"]) == state} == methods
    }
    rows = [row for row in rows if int(row["state_index"]) in completed]
    with h5py.File(MAT_PATH, "r") as handle:
        for state in range(len(dataset.state_ids)):
            if state in completed:
                continue
            snapshots = dataset.read_session(state)
            scenes = list(map(int, dataset.target_scenes[state]))
            covariances = np.stack([sample_covariance(snapshots[scene]) for scene in scenes])
            predicted_counts = np.asarray([shared_counts[(state, scene)] for scene in scenes], np.int64)
            for index, scene in enumerate(scenes):
                covariance = covariances[index]
                mdl_count = source_count_mdl(covariance, snapshots[scene].shape[1])
                predicted_count = int(predicted_counts[index])
                estimates = {
                    "mdl_music": music_2d(covariance, mdl_count, array),
                    "music": music_2d(covariance, predicted_count, array),
                }
                fallback = 0
                try:
                    estimates["esprit"] = l_shaped_esprit(covariance, predicted_count, array)
                except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError):
                    estimates["esprit"] = estimates["music"]
                    fallback = 1
                for method, estimate in estimates.items():
                    rows.append(metric_row(dataset, handle, state, scene, estimate, method, fallback if method == "esprit" else 0))
            rows.sort(key=lambda row: (int(row["state_index"]), str(row["method"]), int(row["scene"])))
            write_csv(scene_path, rows)
            if (state + 1) % 10 == 0 or state + 1 == len(dataset.state_ids):
                print(f"classical/SBL baselines: {state + 1}/{len(dataset.state_ids)} sessions")
    write_csv(scene_path, rows)
    write_csv(OUTPUT_DIRECTORY / "method_summary.csv", summarize(rows, ("method",)))
    write_csv(OUTPUT_DIRECTORY / "breakdown.csv", summarize(rows, ("method", "case")))
    fallback_rows = [{"method": "esprit", "fallback_scenes": sum(int(row["esprit_music_fallback"]) for row in rows if row["method"] == "esprit"), "total_scenes": sum(row["method"] == "esprit" for row in rows)}]
    write_csv(OUTPUT_DIRECTORY / "esprit_fallback_summary.csv", fallback_rows)
    for method in ("mdl_music", "music", "esprit"):
        save_standard_reports([row for row in rows if row["method"] == method], OUTPUT_DIRECTORY / method)


if __name__ == "__main__":
    main()
