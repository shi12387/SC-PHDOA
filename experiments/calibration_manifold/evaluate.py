"""Evaluate physical calibration parameters and manifold reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.dataset import read_calibration
from sc_phdoa.evaluation import (
    CASE_NAMES,
    MANIFOLD_EVALUATION_GRID,
    read_csv,
    write_csv,
)
from sc_phdoa.inference import SCPHDOAInferencePipeline
from sc_phdoa.manifold import decode_calibration, make_manifold
from sc_phdoa.metrics import (
    calibration_component_errors,
    manifold_error_and_coherence,
)
from sc_phdoa.protocol import dataset_path, output_path, protocol_fingerprint

MAT_PATH = dataset_path("external_test")
OUTPUT_DIRECTORY = output_path("calibration_manifold")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def physical_metrics(
    estimate: tuple[np.ndarray, np.ndarray, np.ndarray],
    oracle: tuple[np.ndarray, np.ndarray, np.ndarray],
    pipeline: SCPHDOAInferencePipeline,
) -> tuple[float, float, float, float, float]:
    dataset = pipeline.backend.dataset
    gain, position, coupling = calibration_component_errors(
        estimate, oracle, dataset.spacing, dataset.coupling_pairs
    )
    estimate_manifold = make_manifold(estimate, dataset.wavelength)(
        MANIFOLD_EVALUATION_GRID
    )
    oracle_manifold = make_manifold(oracle, dataset.wavelength)(
        MANIFOLD_EVALUATION_GRID
    )
    manifold_error, coherence = manifold_error_and_coherence(
        estimate_manifold, oracle_manifold
    )
    return gain, position, coupling, manifold_error, coherence


def summarize(rows: list[dict], fields: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[field] for field in fields), []).append(row)
    metrics = (
        "gain_evm",
        "position_rmse_d",
        "coupling_rmse",
        "manifold_error",
        "manifold_coherence",
    )
    return [
        {
            **dict(zip(fields, key)),
            "sessions": len(selected),
            **{
                f"mean_{metric}": float(np.mean([row[metric] for row in selected]))
                for metric in metrics
            },
        }
        for key, selected in groups.items()
    ]


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    pipeline = SCPHDOAInferencePipeline.from_files(
        MAT_PATH, device=DEVICE, validate_identity=True
    )
    dataset = pipeline.backend.dataset
    scene_path = OUTPUT_DIRECTORY / "session_details.csv"
    rows = read_csv(scene_path) if scene_path.is_file() else []
    completed = {
        state for state in range(pipeline.state_count)
        if sum(int(row["state_index"]) == state for row in rows) == 2
    }
    rows = [row for row in rows if int(row["state_index"]) in completed]
    with h5py.File(MAT_PATH, "r") as handle:
        for state in range(pipeline.state_count):
            if state in completed:
                continue
            prediction = pipeline.predict_session(
                state,
                calibration_route="session",
                reuse_preprocessing=True,
            )
            oracle_index = int(dataset.original_indices[state, 0])
            oracle = read_calibration(handle, oracle_index, dataset.sample_count)
            estimates = {
                "nominal": (
                    np.ones(7, np.complex128),
                    np.eye(7, dtype=np.complex128),
                    dataset.nominal_coordinates.copy(),
                ),
                "session_calibrated": decode_calibration(
                    prediction.arrays["calibration_final"],
                    dataset.nominal_coordinates,
                    dataset.spacing,
                    dataset.coupling_pairs,
                ),
            }
            for method, estimate in estimates.items():
                gain, position, coupling, manifold, coherence = physical_metrics(
                    estimate, oracle, pipeline
                )
                rows.append({
                    "state_index": state,
                    "state_id": int(dataset.state_ids[state]),
                    "case_id": int(dataset.impairment_cases[state]),
                    "case": CASE_NAMES[int(dataset.impairment_cases[state])],
                    "method": method,
                    "gain_evm": gain,
                    "position_rmse_d": position,
                    "coupling_rmse": coupling,
                    "manifold_error": manifold,
                    "manifold_coherence": coherence,
                    "protocol_fingerprint": protocol_fingerprint(),
                })
            if (state + 1) % 10 == 0 or state + 1 == pipeline.state_count:
                write_csv(scene_path, rows)
                print(f"calibration sessions {state + 1}/{pipeline.state_count}")
    write_csv(scene_path, rows)
    write_csv(OUTPUT_DIRECTORY / "overall_summary.csv", summarize(rows, ("method",)))
    write_csv(
        OUTPUT_DIRECTORY / "condition_summary.csv",
        summarize(rows, ("method", "case")),
    )


if __name__ == "__main__":
    main()
