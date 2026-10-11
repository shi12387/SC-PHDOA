"""Measure direct end-to-end session latency and peak CUDA memory."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.inference import SCPHDOAInferencePipeline
from sc_phdoa.evaluation import write_csv
from sc_phdoa.protocol import (
    dataset_path,
    load_protocol,
    output_path,
    protocol_fingerprint,
)

PROTOCOL = load_protocol()
MAT_PATH = dataset_path("external_test")
OUTPUT_DIRECTORY = output_path("runtime_benchmark")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_SESSIONS = int(PROTOCOL["runtime_warmup_sessions"])
MEASURED_SESSIONS = int(PROTOCOL["runtime_measured_sessions"])
PROFILE_SESSIONS = int(PROTOCOL["runtime_profile_sessions"])
TARGETS_PER_SESSION = int(PROTOCOL["target_scenes"])
MODULE_FREQUENCIES = {
    "source_count": "18 calls/session",
    "synchronization_frontend": "18 calls/session",
    "session_calibration": "1 call/session",
    "guard_routing": "1 call/session",
    "target_als": f"{TARGETS_PER_SESSION} calls/session",
}


def synchronize() -> None:
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


def stratified_states(cases: np.ndarray, total: int) -> list[int]:
    labels = sorted(np.unique(cases).tolist())
    if total % len(labels):
        raise ValueError(f"{total} is not divisible by {len(labels)} conditions")
    per_condition = total // len(labels)
    selected = []
    for label in labels:
        candidates = np.flatnonzero(cases == label)
        if len(candidates) < per_condition:
            raise ValueError(f"condition {label} has only {len(candidates)} sessions")
        selected.extend(candidates[:per_condition].tolist())
    return sorted(map(int, selected))


def read_existing_rows(
    path: Path, fingerprint: str, selected_states: set[int]
) -> list[dict]:
    if not path.is_file():
        return []
    import csv

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        loaded = list(csv.DictReader(handle))
    rows = []
    seen = set()
    for row in loaded:
        if row.get("protocol_fingerprint") != fingerprint:
            continue
        state = int(row["state_index"])
        if state not in selected_states or state in seen:
            continue
        seen.add(state)
        rows.append({
            "state_index": state,
            "session_latency_ms": float(row["session_latency_ms"]),
            "amortized_target_latency_ms": float(
                row["amortized_target_latency_ms"]
            ),
            "protocol_fingerprint": fingerprint,
        })
    return sorted(rows, key=lambda item: item["state_index"])


def read_profile_rows(
    path: Path, fingerprint: str, selected_states: set[int]
) -> list[dict]:
    if not path.is_file():
        return []
    import csv

    modules = list(MODULE_FREQUENCIES)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        loaded = list(csv.DictReader(handle))
    rows = []
    seen = set()
    for row in loaded:
        if row.get("protocol_fingerprint") != fingerprint:
            continue
        state = int(row["state_index"])
        if state not in selected_states or state in seen:
            continue
        if any(row.get(module, "") == "" for module in modules):
            continue
        seen.add(state)
        parsed = {"state_index": state, "protocol_fingerprint": fingerprint}
        parsed.update({module: float(row[module]) for module in modules})
        parsed["profiled_total_ms"] = float(row["profiled_total_ms"])
        rows.append(parsed)
    return sorted(rows, key=lambda item: item["state_index"])


def summarize_modules(rows: list[dict]) -> list[dict]:
    summary = []
    for module, frequency in MODULE_FREQUENCIES.items():
        values = np.asarray([row[module] for row in rows], np.float64)
        summary.append({
            "module": module,
            "execution_frequency": frequency,
            "mean_ms_per_session": float(np.mean(values)),
            "std_ms_per_session": float(np.std(values, ddof=1)),
            "median_ms_per_session": float(np.median(values)),
            "p90_ms_per_session": float(np.quantile(values, 0.90)),
        })
    return summary


def main() -> None:
    if not MAT_PATH.is_file():
        raise FileNotFoundError(f"Missing external test set: {MAT_PATH}")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    latency_path = OUTPUT_DIRECTORY / "session_latency.csv"
    profile_path = OUTPUT_DIRECTORY / "module_session_latency.csv"
    module_summary_path = OUTPUT_DIRECTORY / "module_latency_summary.csv"
    report_path = OUTPUT_DIRECTORY / "report.json"
    fingerprint = protocol_fingerprint()
    pipeline = SCPHDOAInferencePipeline.from_files(
        MAT_PATH, device=DEVICE, validate_identity=True
    )
    measured_states = stratified_states(
        pipeline.backend.dataset.impairment_cases, MEASURED_SESSIONS
    )
    profile_states = stratified_states(
        pipeline.backend.dataset.impairment_cases, PROFILE_SESSIONS
    )
    rows = read_existing_rows(latency_path, fingerprint, set(measured_states))
    completed = {int(row["state_index"]) for row in rows}
    resumed_sessions = len(completed)
    if completed:
        print(
            f"resuming runtime benchmark: {len(completed)}/"
            f"{MEASURED_SESSIONS} measured sessions complete"
        )
    previous_peak = 0
    if report_path.is_file():
        try:
            previous_report = json.loads(report_path.read_text(encoding="utf-8"))
            previous_peak = int(previous_report.get("peak_cuda_memory_bytes") or 0)
        except (OSError, json.JSONDecodeError):
            print(f"ignoring incomplete runtime report: {report_path}")

    # Storage throughput is excluded from the paper's online latency.
    for state in measured_states[:WARMUP_SESSIONS]:
        snapshots = pipeline.backend.read_session(state)
        pipeline.predict_session(state, snapshots_by_scene=snapshots)
    synchronize()
    if DEVICE.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for position, state in enumerate(measured_states, 1):
        if state in completed:
            continue
        snapshots = pipeline.backend.read_session(state)
        synchronize()
        started = time.perf_counter()
        pipeline.predict_session(state, snapshots_by_scene=snapshots)
        synchronize()
        milliseconds = (time.perf_counter() - started) * 1000.0
        rows.append({
            "state_index": state,
            "session_latency_ms": milliseconds,
            "amortized_target_latency_ms": milliseconds / TARGETS_PER_SESSION,
            "protocol_fingerprint": fingerprint,
        })
        rows.sort(key=lambda item: item["state_index"])
        write_csv(latency_path, rows)
        print(
            f"session {position}/{MEASURED_SESSIONS} "
            f"(state {state}): {milliseconds:.3f} ms"
        )

    if len(rows) != MEASURED_SESSIONS:
        raise RuntimeError(
            f"Runtime benchmark incomplete: {len(rows)}/{MEASURED_SESSIONS} sessions"
        )

    profile_rows = read_profile_rows(
        profile_path, fingerprint, set(profile_states)
    )
    profiled = {int(row["state_index"]) for row in profile_rows}
    if profiled:
        print(
            f"resuming module profile: {len(profiled)}/"
            f"{PROFILE_SESSIONS} sessions complete"
        )
    for position, state in enumerate(profile_states, 1):
        if state in profiled:
            continue
        snapshots = pipeline.backend.read_session(state)
        profile: dict[str, float] = {}
        pipeline.predict_session(
            state,
            snapshots_by_scene=snapshots,
            profile=profile,
        )
        missing = set(MODULE_FREQUENCIES).difference(profile)
        if missing:
            raise RuntimeError(f"Missing module timings: {sorted(missing)}")
        profile_rows.append({
            "state_index": state,
            **{module: profile[module] for module in MODULE_FREQUENCIES},
            "profiled_total_ms": float(sum(profile.values())),
            "protocol_fingerprint": fingerprint,
        })
        profile_rows.sort(key=lambda item: item["state_index"])
        write_csv(profile_path, profile_rows)
        print(
            f"profile {position}/{PROFILE_SESSIONS} "
            f"(state {state}) complete"
        )
    if len(profile_rows) != PROFILE_SESSIONS:
        raise RuntimeError(
            f"Module profile incomplete: {len(profile_rows)}/"
            f"{PROFILE_SESSIONS} sessions"
        )
    module_summary = summarize_modules(profile_rows)
    write_csv(profile_path, profile_rows)
    write_csv(module_summary_path, module_summary)

    latencies = np.asarray([row["session_latency_ms"] for row in rows])
    current_peak = (
        int(torch.cuda.max_memory_allocated()) if DEVICE.startswith("cuda") else 0
    )
    report = {
        "device": DEVICE,
        "warmup_sessions": WARMUP_SESSIONS,
        "measured_sessions": MEASURED_SESSIONS,
        "profiled_sessions": PROFILE_SESSIONS,
        "resumed_sessions": resumed_sessions,
        "targets_per_session": TARGETS_PER_SESSION,
        "measured_state_indices": measured_states,
        "profiled_state_indices": profile_states,
        "mean_session_latency_ms": float(np.mean(latencies)),
        "std_session_latency_ms": float(np.std(latencies, ddof=1)),
        "median_session_latency_ms": float(np.median(latencies)),
        "p90_session_latency_ms": float(np.quantile(latencies, 0.90)),
        "mean_amortized_target_latency_ms": float(
            np.mean(latencies) / TARGETS_PER_SESSION
        ),
        "session_calibration_parameters": sum(
            parameter.numel()
            for parameter in pipeline.model.components.session_calibrator.parameters()
        ),
        "peak_cuda_memory_bytes": (
            max(previous_peak, current_peak) if DEVICE.startswith("cuda") else None
        ),
        "als_iterations": 1,
        "final_iq_refiner": False,
        "disk_io_included": False,
        "timing_scope": (
            "18-scene source count and synchronization; context calibration; "
            "guard routing; six target ALS calls"
        ),
        "module_latency_summary": str(module_summary_path),
        "protocol_fingerprint": fingerprint,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": (
            torch.cuda.get_device_name(0) if DEVICE.startswith("cuda") else None
        ),
    }
    write_csv(latency_path, rows)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    temporary_report.replace(report_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
