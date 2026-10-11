"""Shared evaluation and session-bootstrap reporting for paper experiments."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np

from .dataset import read_calibration, read_doas
from .inference import SCPHDOAInferencePipeline
from .manifold import make_manifold, nominal_manifold
from .metrics import manifold_error_and_coherence, matched_errors, ospa
from .protocol import protocol_fingerprint

MANIFOLD_AZIMUTH_GRID = np.arange(-60.0, 60.0 + 1e-6, 2.0)
MANIFOLD_ELEVATION_GRID = np.arange(5.0, 80.0 + 1e-6, 3.0)
MANIFOLD_EVALUATION_GRID = np.array(
    np.meshgrid(
        MANIFOLD_AZIMUTH_GRID,
        MANIFOLD_ELEVATION_GRID,
        indexing="ij",
    ),
    dtype=np.float64,
).reshape(2, -1).T

STANDARD_ROW_FIELDS = {
    "state_index", "scene", "K_true", "K_predicted", "count_correct",
    "route", "ospa", "scene_rmse", "matched_squared_error_sum",
    "matched_source_count", "maximum_matched_error", "e2e_1deg",
    "e2e_2deg", "e2e_3deg", "manifold_error", "manifold_coherence",
    "protocol_fingerprint",
}

CASE_NAMES = {
    0: "ideal",
    1: "gain_phase",
    2: "coupling",
    3: "position",
    4: "compound",
}

INTEGER_FIELDS = {
    "state_index", "state_id", "calibration_source_state", "scene",
    "original_index", "case_id",
    "K_true", "K_predicted", "count_correct", "matched_source_count",
    "e2e_1deg", "e2e_2deg", "e2e_3deg",
}
FLOAT_FIELDS = {
    "snr", "ospa", "scene_rmse", "matched_squared_error_sum",
    "maximum_matched_error", "manifold_error", "manifold_coherence",
    "gain_evm", "position_rmse_d", "coupling_rmse", "runtime_ms",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    """Read an evaluation CSV and restore the standard numeric field types."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in INTEGER_FIELDS.intersection(row):
            row[field] = int(row[field])
        for field in FLOAT_FIELDS.intersection(row):
            row[field] = float(row[field])
    return rows


def load_completed_evaluation(
    path: Path,
    expected_rows: int,
    expected_states: int,
) -> list[dict] | None:
    """Return a complete cached evaluation, or None for missing/partial output."""
    if not path.is_file():
        return None
    rows = read_csv(path)
    if rows and (
        not STANDARD_ROW_FIELDS.issubset(rows[0])
        or any(
            row.get("protocol_fingerprint") != protocol_fingerprint()
            for row in rows
        )
    ):
        print(f"ignoring incompatible cache {path}")
        return None
    state_count = len({int(row["state_index"]) for row in rows}) if rows else 0
    if len(rows) != expected_rows or state_count != expected_states:
        print(
            f"ignoring incomplete cache {path}: rows={len(rows)}/{expected_rows}, "
            f"states={state_count}/{expected_states}"
        )
        return None
    print(f"reusing completed evaluation: {path}")
    return rows


def evaluate_sessions(
    mat_path: Path,
    output_directory: Path,
    device: str,
    states: Iterable[int] | None = None,
    validate_identity: bool = False,
    calibration_route: str = "guard",
    context_scenes_by_state: dict[int, np.ndarray] | None = None,
) -> list[dict]:
    output_directory.mkdir(parents=True, exist_ok=True)
    pipeline = SCPHDOAInferencePipeline.from_files(
        mat_path, device=device, validate_identity=validate_identity
    )
    return evaluate_pipeline(
        pipeline,
        mat_path,
        output_directory,
        states=states,
        calibration_route=calibration_route,
        context_scenes_by_state=context_scenes_by_state,
    )


def evaluate_pipeline(
    pipeline: SCPHDOAInferencePipeline,
    mat_path: Path,
    output_directory: Path,
    states: Iterable[int] | None = None,
    calibration_route: str = "guard",
    context_scenes_by_state: dict[int, np.ndarray] | None = None,
    save_predictions: bool = False,
    reuse_preprocessing: bool = False,
    calibration_source_by_state: dict[int, int] | None = None,
    target_counts_by_state: dict[int, dict[int, int]] | None = None,
    manifold_by_state: dict[int, dict] | None = None,
) -> list[dict]:
    """Evaluate a pipeline with atomic, session-level checkpointing."""
    output_directory.mkdir(parents=True, exist_ok=True)
    selected_states = list(range(pipeline.state_count)) if states is None else list(states)
    dataset = pipeline.backend.dataset
    scene_path = output_directory / "scene_details.csv"
    rows = read_csv(scene_path) if scene_path.is_file() else []
    fingerprint = protocol_fingerprint()
    if rows and (
        not STANDARD_ROW_FIELDS.issubset(rows[0])
        or any(row.get("protocol_fingerprint") != fingerprint for row in rows)
    ):
        print(f"discarding incompatible partial cache: {scene_path}")
        rows = []
    selected_set = set(map(int, selected_states))
    rows = [row for row in rows if int(row["state_index"]) in selected_set]
    expected_by_state = {
        int(state): len(dataset.target_scenes[int(state)]) for state in selected_states
    }
    state_row_counts = {
        state: sum(int(row["state_index"]) == state for row in rows)
        for state in selected_states
    }
    completed = {
        state for state, count in state_row_counts.items()
        if count == expected_by_state[state]
    }
    rows = [row for row in rows if int(row["state_index"]) in completed]
    if completed:
        print(
            f"resuming {output_directory.name}: "
            f"{len(completed)}/{len(selected_states)} sessions complete"
        )
    with h5py.File(mat_path, "r") as handle:
        for position, state in enumerate(selected_states, 1):
            state = int(state)
            if state in completed:
                continue
            context = (
                None
                if context_scenes_by_state is None
                else context_scenes_by_state[state]
            )
            prediction = pipeline.predict_session(
                state,
                context_scenes=context,
                calibration_route=calibration_route,
                reuse_preprocessing=reuse_preprocessing,
                calibration_source_state=(
                    None
                    if calibration_source_by_state is None
                    else calibration_source_by_state[state]
                ),
                target_count_override=(
                    None
                    if target_counts_by_state is None
                    else target_counts_by_state[state]
                ),
                manifold_override=(
                    None
                    if manifold_by_state is None
                    else manifold_by_state[state]
                ),
            )
            if save_predictions:
                prediction.save(
                    output_directory / "predictions" / f"session_{state:04d}"
                )
            calibration_index = int(dataset.original_indices[state, 0])
            oracle_calibration = read_calibration(
                handle, calibration_index, dataset.sample_count
            )
            oracle_values = make_manifold(
                oracle_calibration, dataset.wavelength
            )(MANIFOLD_EVALUATION_GRID)
            if prediction.calibration_route == "oracle":
                if manifold_by_state is None:
                    raise RuntimeError("oracle route is missing its manifold")
                selected_manifold = manifold_by_state[state]["manifold"]
            elif prediction.calibration_route == "nominal":
                selected_manifold = nominal_manifold(
                    dataset.nominal_coordinates, dataset.wavelength
                )
            else:
                selected_vector = (
                    prediction.arrays["corrected_calibration"]
                    if prediction.calibration_route == "residual"
                    else prediction.arrays["calibration_final"]
                )
                selected_manifold = pipeline.backend._decode_manifold(
                    selected_vector
                )
            manifold_error, manifold_coherence = manifold_error_and_coherence(
                selected_manifold(MANIFOLD_EVALUATION_GRID), oracle_values
            )
            for target in prediction.targets:
                scene = int(target.scene_index)
                original = int(dataset.original_indices[state, scene])
                true_count = int(dataset.source_counts[state, scene])
                estimated_count = int(target.source_count)
                truth = read_doas(handle, original, dataset.sample_count, true_count)
                estimate = np.asarray(target.doas_deg[:estimated_count], np.float64)
                count_correct = estimated_count == true_count
                errors = matched_errors(estimate, truth) if count_correct else np.asarray([])
                maximum = float(np.max(errors)) if len(errors) else float("nan")
                row = {
                    "state_index": state,
                    "calibration_source_state": int(
                        prediction.arrays["calibration_source_state"]
                    ),
                    "state_id": int(dataset.state_ids[state]),
                    "scene": scene,
                    "original_index": original,
                    "case_id": int(dataset.impairment_cases[state]),
                    "case": CASE_NAMES.get(int(dataset.impairment_cases[state]), str(int(dataset.impairment_cases[state]))),
                    "snr": float(dataset.snrs[state, scene]),
                    "K_true": true_count,
                    "K_predicted": estimated_count,
                    "count_correct": int(count_correct),
                    "route": prediction.calibration_route,
                    "ospa": ospa(estimate, truth),
                    "scene_rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else float("nan"),
                    "matched_squared_error_sum": float(np.sum(errors**2)) if len(errors) else 0.0,
                    "matched_source_count": int(len(errors)),
                    "maximum_matched_error": maximum,
                    "e2e_1deg": int(count_correct and maximum <= 1.0),
                    "e2e_2deg": int(count_correct and maximum <= 2.0),
                    "e2e_3deg": int(count_correct and maximum <= 3.0),
                    "manifold_error": manifold_error,
                    "manifold_coherence": manifold_coherence,
                    "protocol_fingerprint": fingerprint,
                }
                rows.append(row)
            rows.sort(key=lambda row: (int(row["state_index"]), int(row["scene"])))
            write_csv(scene_path, rows)
            if position % 10 == 0 or position == len(selected_states):
                print(f"evaluated sessions {position}/{len(selected_states)}")
    write_csv(scene_path, rows)
    return rows


def summarize(rows: list[dict], group_fields: tuple[str, ...] = ()) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, selected in grouped.items():
        matched_squared = sum(float(row["matched_squared_error_sum"]) for row in selected)
        matched_sources = sum(int(row["matched_source_count"]) for row in selected)
        scene_rmse = np.asarray([row["scene_rmse"] for row in selected], np.float64)
        scene_rmse = scene_rmse[np.isfinite(scene_rmse)]
        summary = {
            **dict(zip(group_fields, key)),
            "sessions": len({int(row["state_index"]) for row in selected}),
            "scenes": len(selected),
            "count_accuracy": float(np.mean([row["count_correct"] for row in selected])),
            "mean_ospa": float(np.mean([row["ospa"] for row in selected])),
            "pooled_correct_k_rmse": (
                float(np.sqrt(matched_squared / matched_sources)) if matched_sources else float("nan")
            ),
            "mean_scene_correct_k_rmse": (
                float(np.mean(scene_rmse)) if len(scene_rmse) else float("nan")
            ),
            "e2e_1deg": float(np.mean([row["e2e_1deg"] for row in selected])),
            "e2e_2deg": float(np.mean([row["e2e_2deg"] for row in selected])),
            "e2e_3deg": float(np.mean([row["e2e_3deg"] for row in selected])),
            "residual_route_rate": float(np.mean([row["route"] == "residual" for row in selected])),
        }
        if all("manifold_error" in row for row in selected):
            manifold_error = np.asarray([row["manifold_error"] for row in selected], np.float64)
            manifold_coherence = np.asarray([row["manifold_coherence"] for row in selected], np.float64)
            summary["manifold_error"] = (
                float(np.mean(manifold_error[np.isfinite(manifold_error)]))
                if np.any(np.isfinite(manifold_error)) else float("nan")
            )
            summary["manifold_coherence"] = (
                float(np.mean(manifold_coherence[np.isfinite(manifold_coherence)]))
                if np.any(np.isfinite(manifold_coherence)) else float("nan")
            )
        output.append(summary)
    return output


def session_bootstrap(
    rows: list[dict], repeats: int = 5000, seed: int = 2026
) -> list[dict]:
    state_ids = np.asarray(sorted({int(row["state_index"]) for row in rows}), np.int64)
    by_state = {state: [row for row in rows if int(row["state_index"]) == state] for state in state_ids}
    rng = np.random.default_rng(seed)
    metrics = {
        "mean_ospa": [],
        "mean_scene_correct_k_rmse": [],
        "e2e_3deg": [],
        "count_accuracy": [],
    }
    for _ in range(repeats):
        sampled = rng.choice(state_ids, size=len(state_ids), replace=True)
        selected = [row for state in sampled for row in by_state[int(state)]]
        metrics["mean_ospa"].append(float(np.mean([row["ospa"] for row in selected])))
        correct_k_rmse = np.asarray(
            [row["scene_rmse"] for row in selected], np.float64
        )
        correct_k_rmse = correct_k_rmse[np.isfinite(correct_k_rmse)]
        metrics["mean_scene_correct_k_rmse"].append(
            float(np.mean(correct_k_rmse)) if len(correct_k_rmse) else float("nan")
        )
        metrics["e2e_3deg"].append(float(np.mean([row["e2e_3deg"] for row in selected])))
        metrics["count_accuracy"].append(float(np.mean([row["count_correct"] for row in selected])))
    return [
        {
            "metric": name,
            "estimate": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "bootstrap_repeats": repeats,
        }
        for name, values in metrics.items()
    ]


def paired_session_bootstrap(
    rows: list[dict],
    method_a: str,
    method_b: str,
    repeats: int = 5000,
    seed: int = 2026,
) -> list[dict]:
    """Bootstrap paired session differences as method_b minus method_a."""
    selected = {
        method: [row for row in rows if row.get("method") == method]
        for method in (method_a, method_b)
    }
    state_sets = [
        {int(row["state_index"]) for row in method_rows}
        for method_rows in selected.values()
    ]
    if not state_sets or state_sets[0] != state_sets[1]:
        raise ValueError(f"Methods are not paired by session: {method_a}, {method_b}")
    states = np.asarray(sorted(state_sets[0]), np.int64)
    metrics = {
        "mean_ospa": "ospa",
        "mean_scene_correct_k_rmse": "scene_rmse",
        "manifold_error": "manifold_error",
        "e2e_3deg": "e2e_3deg",
        "count_accuracy": "count_correct",
    }
    per_state: dict[str, dict[str, np.ndarray]] = {}
    for summary_name, row_name in metrics.items():
        per_state[summary_name] = {}
        for method, method_rows in selected.items():
            grouped = {
                state: [
                    float(row[row_name]) for row in method_rows
                    if int(row["state_index"]) == state
                ]
                for state in states
            }
            per_state[summary_name][method] = np.asarray([
                (
                    float(np.mean(finite))
                    if len(finite := np.asarray(grouped[state], np.float64)[
                        np.isfinite(np.asarray(grouped[state], np.float64))
                    ])
                    else float("nan")
                )
                for state in states
            ])
    rng = np.random.default_rng(seed)
    output = []
    for metric, values in per_state.items():
        if not all(np.any(np.isfinite(value)) for value in values.values()):
            continue
        differences = values[method_b] - values[method_a]
        differences = differences[np.isfinite(differences)]
        if not len(differences):
            continue
        bootstrap = np.empty(repeats, np.float64)
        for index in range(repeats):
            sampled = rng.integers(0, len(differences), size=len(differences))
            bootstrap[index] = float(np.mean(differences[sampled]))
        output.append({
            "method_a": method_a,
            "method_b": method_b,
            "metric": metric,
            "difference_b_minus_a": float(np.mean(differences)),
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
            "bootstrap_repeats": repeats,
        })
    return output


def save_standard_reports(rows: list[dict], output_directory: Path) -> None:
    overall = summarize(rows)
    by_source_count = summarize(rows, ("K_true",))
    by_snr = summarize(rows, ("snr",))
    by_condition = summarize(rows, ("case",))
    bootstrap = session_bootstrap(rows)
    write_csv(output_directory / "overall_summary.csv", overall)
    write_csv(output_directory / "source_count_summary.csv", by_source_count)
    write_csv(output_directory / "snr_summary.csv", by_snr)
    write_csv(output_directory / "array_condition_summary.csv", by_condition)
    write_csv(output_directory / "session_bootstrap.csv", bootstrap)
    report = {
        "overall": overall[0],
        "source_count": by_source_count,
        "snr": by_snr,
        "array_condition": by_condition,
        "session_bootstrap": bootstrap,
    }
    (output_directory / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )
