"""Verify completeness and numerical parity of paper experiment outputs."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.protocol import output_path, protocol_fingerprint

EXPECTED = {
    "overall_comparison/scene_details.csv": 5400,
    "overall_comparison/method_summary.csv": 3,
    "classical_baselines/scene_details.csv": 5400,
    "classical_baselines/method_summary.csv": 3,
    "em_sbl/scene_details.csv": 1800,
    "em_sbl/method_summary.csv": 1,
    "observation_budget/scene_details.csv": 7200,
    "observation_budget/method_summary.csv": 4,
    "assembled_paper_results/E01_scene_details.csv": 14400,
    "assembled_paper_results/E01_overall_comparison.csv": 8,
    "assembled_paper_results/E06_snr_robustness.csv": 24,
    "assembled_paper_results/E07_source_count_breakdown.csv": 24,
    "assembled_paper_results/E09_array_condition_breakdown.csv": 40,
    "calibration_manifold/session_details.csv": 600,
    "calibration_manifold/overall_summary.csv": 2,
    "oracle_calibration/scene_details.csv": 7200,
    "oracle_calibration/method_summary.csv": 4,
    "oracle_source_count/scene_details.csv": 3600,
    "oracle_source_count/method_summary.csv": 2,
    "context_sensitivity/scene_details.csv": 73800,
    "context_sensitivity/repeat_summary.csv": 41,
    "context_sensitivity/context_summary.csv": 5,
    "ood_generalization/scene_details.csv": 16200,
    "ood_generalization/distribution_summary.csv": 9,
    "close_source_resolution/scene_details.csv": 7200,
    "close_source_resolution/resolution_summary.csv": 60,
    "calibration_drift/scene_details.csv": 28800,
    "calibration_drift/drift_summary.csv": 80,
    "shuffled_session/scene_details.csv": 3600,
    "shuffled_session/method_summary.csv": 2,
    "calibration_input_ablation_evaluation/scene_details.csv": 5400,
    "calibration_input_ablation_evaluation/variant_summary.csv": 3,
    "runtime_benchmark/session_latency.csv": 60,
    "runtime_benchmark/module_session_latency.csv": 20,
    "runtime_benchmark/module_latency_summary.csv": 5,
    "als_depth_validation/scene_details.csv": 4800,
    "als_depth_validation/summary.csv": 4,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    failures: list[str] = []
    checks = []
    outputs = ROOT / "outputs"
    for relative, expected_rows in EXPECTED.items():
        path = outputs / relative
        if not path.is_file():
            failures.append(f"missing output: {relative}")
            checks.append({"file": relative, "ok": False, "reason": "missing"})
            continue
        rows = read_csv(path)
        ok = len(rows) == expected_rows
        if not ok:
            failures.append(
                f"{relative}: rows={len(rows)}, expected={expected_rows}"
            )
        checks.append({
            "file": relative, "ok": ok, "rows": len(rows),
            "expected_rows": expected_rows,
        })

    reference = json.loads(
        (ROOT / "reference_results" / "final_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    tolerance = float(reference["tolerance"])
    summary_path = output_path("overall_comparison") / "method_summary.csv"
    if summary_path.is_file():
        actual = {row["method"]: row for row in read_csv(summary_path)}
        for method, expected_metrics in reference["overall"].items():
            if method not in actual:
                failures.append(f"overall summary missing method: {method}")
                continue
            for metric, expected_value in expected_metrics.items():
                value = float(actual[method][metric])
                if not math.isclose(value, expected_value, abs_tol=tolerance):
                    failures.append(
                        f"overall/{method}/{metric}: {value} != {expected_value}"
                    )

    sbl_path = output_path("em_sbl") / "method_summary.csv"
    if sbl_path.is_file():
        actual = {row["method"]: row for row in read_csv(sbl_path)}
        if "em_sbl" not in actual:
            failures.append("EM-SBL summary missing em_sbl")
        else:
            expected = reference["em_sbl"]
            for metric in ("mean_ospa", "mean_scene_correct_k_rmse", "e2e_3deg"):
                value = float(actual["em_sbl"][metric])
                if not math.isclose(value, expected[metric], abs_tol=expected["tolerance"]):
                    failures.append(f"em_sbl/{metric}: {value} != {expected[metric]}")

    context_path = output_path("context_sensitivity") / "context_summary.csv"
    if context_path.is_file():
        actual = {str(int(row["context_count"])): row for row in read_csv(context_path)}
        for count, expected_metrics in reference["context_sensitivity"].items():
            if count not in actual:
                failures.append(f"context summary missing Nc={count}")
                continue
            for metric, expected_value in expected_metrics.items():
                value = float(actual[count][metric])
                if not math.isclose(value, expected_value, abs_tol=tolerance):
                    failures.append(
                        f"context/Nc={count}/{metric}: {value} != {expected_value}"
                    )

    details = list(outputs.glob("*/scene_details.csv"))
    fingerprint = protocol_fingerprint()
    for path in details:
        rows = read_csv(path)
        if rows and "protocol_fingerprint" in rows[0]:
            fingerprints = {row["protocol_fingerprint"] for row in rows}
            if fingerprints != {fingerprint}:
                failures.append(f"protocol mismatch in {path.relative_to(outputs)}")

    report = {
        "ok": not failures,
        "protocol_fingerprint": fingerprint,
        "checks": checks,
        "failures": failures,
    }
    destination = output_path("reproduction") / "verification_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
