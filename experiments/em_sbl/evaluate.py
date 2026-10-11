"""Run the frozen E01 EM-SBL baseline with resumable state-level caching."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.baselines import CovarianceSBL, NominalArray
from sc_phdoa.baselines.sbl import normalized_scene_covariance
from sc_phdoa.checkpoints import CheckpointPaths
from sc_phdoa.config import DEFAULT_CONFIG
from sc_phdoa.dataset import SessionDataset, read_doas
from sc_phdoa.evaluation import CASE_NAMES, read_csv, save_standard_reports, summarize, write_csv
from sc_phdoa.metrics import matched_errors, ospa
from sc_phdoa.protocol import dataset_path, output_path, protocol_fingerprint
from sc_phdoa.source_count import SourceCountEstimator

MAT_PATH = dataset_path("external_test")
OUTPUT_DIRECTORY = output_path("em_sbl")
COUNT_DETAILS = output_path("overall_comparison") / "nominal_als" / "scene_details.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METHOD = "em_sbl"
MAX_SOURCES = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_signature() -> str:
    stat = MAT_PATH.stat()
    payload = {
        "protocol": protocol_fingerprint(),
        "mat_path": str(MAT_PATH.resolve()),
        "mat_size": stat.st_size,
        "mat_mtime_ns": stat.st_mtime_ns,
        "source_count_checkpoint": _sha256(CheckpointPaths.bundled().source_count),
        "sbl_implementation": _sha256(ROOT / "src" / "sc_phdoa" / "baselines" / "sbl.py"),
        "grid_step_deg": 2.0,
        "iterations": 12,
        "damping": 0.65,
        "noise_floor": 0.002,
        "nms_deg": 4.0,
        "local_radius_deg": 2.5,
        "local_iterations": 25,
        "local_ridge": 1e-4,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _shared_counts(dataset: SessionDataset) -> dict[tuple[int, int], int]:
    if not COUNT_DETAILS.is_file():
        return {}
    rows = read_csv(COUNT_DETAILS)
    if any(row.get("protocol_fingerprint") != protocol_fingerprint() for row in rows):
        return {}
    if any(
        int(row["original_index"]) != int(dataset.original_indices[int(row["state_index"]), int(row["scene"])])
        or int(row["state_id"]) != int(dataset.state_ids[int(row["state_index"])])
        or not 1 <= int(row["K_predicted"]) <= MAX_SOURCES
        for row in rows
    ):
        return {}
    counts = {
        (int(row["state_index"]), int(row["scene"])): int(row["K_predicted"])
        for row in rows
    }
    expected = sum(len(scenes) for scenes in dataset.target_scenes)
    if len(counts) != expected:
        return {}
    return counts


def _load_state_cache(
    path: Path, signature: str, state: int, state_id: int, scenes: np.ndarray,
    shared_counts: dict[tuple[int, int], int],
) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as stored:
            cache = {key: stored[key] for key in stored.files}
        if (
            str(cache["signature"].item()) != signature
            or int(cache["state_index"]) != state
            or int(cache["state_id"]) != state_id
            or not np.array_equal(cache["scenes"], scenes)
            or cache["angles"].shape != (len(scenes), MAX_SOURCES, 2)
            or cache["counts"].shape != (len(scenes),)
            or np.any((cache["counts"] < 1) | (cache["counts"] > MAX_SOURCES))
        ):
            return None
        if any(
            not np.all(np.isfinite(cache["angles"][index, :int(count)]))
            for index, count in enumerate(cache["counts"])
        ):
            return None
        if shared_counts and not np.array_equal(
            cache["counts"],
            [shared_counts[(state, int(scene))] for scene in scenes],
        ):
            return None
        return cache
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def _save_state_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **cache)
    os.replace(temporary, path)


def _evaluate_state(
    dataset: SessionDataset,
    state: int,
    scenes: np.ndarray,
    signature: str,
    shared_counts: dict[tuple[int, int], int],
    solver: CovarianceSBL,
    count_model: SourceCountEstimator | None,
) -> tuple[dict[str, np.ndarray], SourceCountEstimator | None]:
    snapshots = dataset.read_session(state)
    if shared_counts:
        counts = np.asarray(
            [shared_counts[(state, int(scene))] for scene in scenes], np.int64
        )
    else:
        if count_model is None:
            count_model = SourceCountEstimator.load(CheckpointPaths.bundled().source_count)
        counts = np.asarray([
            count_model.predict(
                snapshots[int(scene)], DEFAULT_CONFIG.payload_start,
                dataset.preamble, dataset.maximum_delay,
            ).count
            for scene in scenes
        ], np.int64)
    covariance = np.stack([
        normalized_scene_covariance(snapshots[int(scene)]) for scene in scenes
    ])
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    estimates = solver.estimate_batch(covariance, counts)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    runtime_ms = 1000.0 * (time.perf_counter() - started) / len(scenes)
    padded = np.full((len(scenes), MAX_SOURCES, 2), np.nan, np.float32)
    for index, estimate in enumerate(estimates):
        padded[index, :len(estimate)] = estimate
    cache = {
        "signature": np.asarray(signature),
        "state_index": np.asarray(state, np.int64),
        "state_id": np.asarray(dataset.state_ids[state], np.int64),
        "scenes": scenes,
        "counts": counts,
        "angles": padded,
        "runtime_ms": np.asarray(runtime_ms, np.float64),
    }
    return cache, count_model


def _metric_rows(
    dataset: SessionDataset, handle: h5py.File, state: int,
    cache: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for index, scene_value in enumerate(cache["scenes"]):
        scene = int(scene_value)
        original = int(dataset.original_indices[state, scene])
        true_count = int(dataset.source_counts[state, scene])
        count = int(cache["counts"][index])
        estimate = cache["angles"][index, :count]
        truth = read_doas(handle, original, dataset.sample_count, true_count)
        correct = count == true_count
        errors = matched_errors(estimate, truth) if correct else np.asarray([])
        maximum = float(np.max(errors)) if len(errors) else float("nan")
        rows.append({
            "method": METHOD,
            "state_index": state,
            "state_id": int(dataset.state_ids[state]),
            "scene": scene,
            "original_index": original,
            "case_id": int(dataset.impairment_cases[state]),
            "case": CASE_NAMES[int(dataset.impairment_cases[state])],
            "snr": float(dataset.snrs[state, scene]),
            "K_true": true_count,
            "K_predicted": count,
            "count_correct": int(correct),
            "route": "not_applicable",
            "ospa": ospa(estimate, truth),
            "scene_rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else float("nan"),
            "matched_squared_error_sum": float(np.sum(errors ** 2)) if len(errors) else 0.0,
            "matched_source_count": int(len(errors)),
            "maximum_matched_error": maximum,
            "e2e_1deg": int(correct and maximum <= 1.0),
            "e2e_2deg": int(correct and maximum <= 2.0),
            "e2e_3deg": int(correct and maximum <= 3.0),
            "manifold_error": float("nan"),
            "manifold_coherence": float("nan"),
            "runtime_ms": float(cache["runtime_ms"]),
            "protocol_fingerprint": protocol_fingerprint(),
        })
    return rows


def main() -> None:
    if not MAT_PATH.is_file():
        raise FileNotFoundError(f"Missing external test set: {MAT_PATH}")
    dataset = SessionDataset.load_external(MAT_PATH, validate_identity=True)
    signature = _cache_signature()
    shared_counts = _shared_counts(dataset)
    solver = CovarianceSBL(
        NominalArray(dataset.nominal_coordinates, dataset.wavelength, grid_step_deg=2.0),
        DEVICE,
    )
    count_model = None
    rows = []
    cache_directory = OUTPUT_DIRECTORY / "state_cache"
    with h5py.File(MAT_PATH, "r") as handle:
        for state in range(len(dataset.state_ids)):
            scenes = np.asarray(dataset.target_scenes[state], np.int64)
            path = cache_directory / f"state_{state:04d}.npz"
            cache = _load_state_cache(
                path, signature, state, int(dataset.state_ids[state]), scenes, shared_counts
            )
            if cache is None:
                cache, count_model = _evaluate_state(
                    dataset, state, scenes, signature, shared_counts, solver, count_model
                )
                _save_state_cache(path, cache)
                action = "computed"
            else:
                action = "reused"
            rows.extend(_metric_rows(dataset, handle, state, cache))
            if (state + 1) % 10 == 0 or state + 1 == len(dataset.state_ids):
                print(f"EM-SBL: state {state + 1}/{len(dataset.state_ids)} {action}", flush=True)

    write_csv(OUTPUT_DIRECTORY / "scene_details.csv", rows)
    summary = summarize(rows, ("method",))
    write_csv(OUTPUT_DIRECTORY / "method_summary.csv", summary)
    write_csv(OUTPUT_DIRECTORY / "breakdown.csv", summarize(rows, ("method", "case")))
    save_standard_reports(rows, OUTPUT_DIRECTORY / METHOD)
    result = summary[0]
    print(
        f"EM-SBL E01: OSPA={result['mean_ospa']:.3f} deg, "
        f"Correct-K RMSE={result['mean_scene_correct_k_rmse']:.3f} deg, "
        f"E2E@3={100 * result['e2e_3deg']:.2f}%"
    )
    print(f"Saved: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
