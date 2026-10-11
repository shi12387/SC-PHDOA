"""E15: retrospective 0/1/3/5-step ALS check on the original validation split."""

from __future__ import annotations

import sys
import time
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train_session_calibration import load_cache
from sc_phdoa.als import calibrated_als
from sc_phdoa.calibration_training import stratified_state_split
from sc_phdoa.checkpoints import CheckpointPaths, load_components
from sc_phdoa.config import DEFAULT_CONFIG
from sc_phdoa.dataset import read_doas, read_iq
from sc_phdoa.evaluation import CASE_NAMES, summarize, write_csv
from sc_phdoa.manifold import COARSE_GRID, decode_calibration, make_manifold
from sc_phdoa.metrics import matched_errors, ospa
from sc_phdoa.protocol import dataset_path, output_path, protocol_fingerprint
from sc_phdoa.synchronization import estimate_synchronization_and_mixing

MAT_PATH = dataset_path("session_calibration_training")
CACHE_DIRECTORY = output_path("session_calibration_cache")
OUTPUT_DIRECTORY = output_path("als_depth_validation")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALS_STEPS = (0, 1, 3, 5)
SEED = 2026


def scene_partitions(states: np.ndarray, scene_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    context = np.zeros((len(states), 8), np.int64)
    guard = np.zeros((len(states), 4), np.int64)
    target = np.zeros((len(states), 6), np.int64)
    for row, state in enumerate(states):
        order = np.random.default_rng(SEED + 104729 * int(state)).permutation(scene_count)
        context[row], guard[row], target[row] = np.sort(order[:8]), np.sort(order[8:12]), np.sort(order[12:])
    return context, guard, target


def main() -> None:
    arrays = load_cache(MAT_PATH, CACHE_DIRECTORY)
    splits = stratified_state_split(arrays["state_cases"], SEED)
    if {name: len(values) for name, values in splits.items()} != {"train": 600, "validation": 200, "test": 200}:
        raise RuntimeError("Unexpected session-calibration data split")
    states = np.asarray(splits["validation"], np.int64)
    contexts, _, targets = scene_partitions(states, arrays["features"].shape[1])
    components = load_components(CheckpointPaths.bundled(), DEVICE)
    rows = []
    scene_path = OUTPUT_DIRECTORY / "scene_details.csv"
    if scene_path.is_file():
        from sc_phdoa.evaluation import read_csv
        rows = read_csv(scene_path)
    complete_states = {state for state in states if sum(int(row["state_index"]) == int(state) for row in rows) == 24}
    rows = [row for row in rows if int(row["state_index"]) in complete_states]
    pairs = [tuple(map(int, pair)) for pair in arrays["pairs"]]
    with h5py.File(MAT_PATH, "r") as handle:
        sample_count = int(np.prod(np.asarray(handle["num_sources_all"]).shape))
        preamble = np.asarray(handle["shared_preamble_real"]).reshape(-1) + 1j * np.asarray(handle["shared_preamble_imag"]).reshape(-1)
        preamble = preamble[:256]
        maximum_delay = int(np.asarray(handle["max_timing_offset"]).reshape(-1)[0])
        for position, state_value in enumerate(states):
            state = int(state_value)
            if state in complete_states:
                continue
            context = contexts[position]
            analytic = torch.from_numpy(arrays["features"][state, context][None]).to(DEVICE)
            mask = torch.from_numpy(arrays["source_mask"][state, context][None]).to(DEVICE)
            aligned = torch.from_numpy(np.asarray(arrays["aligned_iq"][state, context], np.float32)[None]).to(DEVICE)
            with torch.inference_mode():
                calibration = components.session_calibrator(analytic, mask, aligned)["final"][0].cpu().numpy()
            decoded = decode_calibration(calibration, arrays["nominal"], float(arrays["spacing"]), pairs)
            manifold = make_manifold(decoded, float(arrays["wavelength"]))
            grid = manifold(COARSE_GRID)
            state_rows = []
            for scene_value in targets[position]:
                scene = int(scene_value)
                original = int(arrays["originals"][state, scene])
                snapshots = read_iq(handle, original, sample_count)
                count_prediction = components.source_count.predict(snapshots, DEFAULT_CONFIG.payload_start, preamble, maximum_delay)
                predicted_count = count_prediction.count
                mixing, _ = estimate_synchronization_and_mixing(snapshots, preamble, maximum_delay, predicted_count)
                true_count = int(arrays["counts"][state, scene])
                truth = read_doas(handle, original, sample_count, true_count)
                for steps in ALS_STEPS:
                    config = replace(DEFAULT_CONFIG, als_iterations=steps)
                    started = time.perf_counter()
                    _, estimate, _ = calibrated_als(snapshots, mixing, manifold, grid, config)
                    runtime_ms = 1000.0 * (time.perf_counter() - started)
                    correct = predicted_count == true_count
                    errors = matched_errors(estimate, truth) if correct else np.asarray([])
                    maximum = float(np.max(errors)) if len(errors) else float("nan")
                    state_rows.append({
                        "method": f"session_calibrated_als_{steps}", "als_steps": steps,
                        "state_index": state, "state_id": int(arrays["state_ids"][state]),
                        "scene": scene, "case": CASE_NAMES[int(arrays["state_cases"][state])],
                        "snr": float(arrays["snrs"][state, scene]), "K_true": true_count,
                        "K_predicted": predicted_count, "count_correct": int(correct), "route": "session",
                        "ospa": ospa(estimate, truth),
                        "scene_rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else float("nan"),
                        "matched_squared_error_sum": float(np.sum(errors ** 2)) if len(errors) else 0.0,
                        "matched_source_count": int(len(errors)), "maximum_matched_error": maximum,
                        "e2e_1deg": int(correct and maximum <= 1.0), "e2e_2deg": int(correct and maximum <= 2.0),
                        "e2e_3deg": int(correct and maximum <= 3.0), "runtime_ms": runtime_ms,
                        "manifold_error": float("nan"), "manifold_coherence": float("nan"),
                        "protocol_fingerprint": protocol_fingerprint(),
                    })
            rows.extend(state_rows)
            write_csv(scene_path, rows)
            if (position + 1) % 10 == 0 or position + 1 == len(states):
                print(f"validation ALS depth: {position + 1}/{len(states)} sessions")
    summary = summarize(rows, ("method", "als_steps"))
    for result in summary:
        selected = [row["runtime_ms"] for row in rows if row["method"] == result["method"]]
        result["mean_als_runtime_ms"] = float(np.mean(selected))
        result["median_als_runtime_ms"] = float(np.median(selected))
    write_csv(OUTPUT_DIRECTORY / "summary.csv", summary)
    write_csv(OUTPUT_DIRECTORY / "breakdown.csv", summarize(rows, ("method", "als_steps", "case")))
    (OUTPUT_DIRECTORY / "protocol.json").write_text(json.dumps({
        "selection_partition": "original_200_state_validation_split",
        "external_test_accessed": False,
        "guard_routing": False,
        "final_iq_refiner": False,
        "candidate_als_iterations": list(ALS_STEPS),
        "frozen_final_als_iterations": 1,
        "selection_metrics": ["mean_ospa", "e2e_3deg", "mean_als_runtime_ms"],
        "protocol_fingerprint": protocol_fingerprint(),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
