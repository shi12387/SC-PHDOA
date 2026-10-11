"""Assemble E01/E06/E07/E09 tables from shared scene-level inference."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.evaluation import read_csv, summarize, write_csv
from sc_phdoa.protocol import output_path

DESTINATION = output_path("assembled_paper_results")


def require(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prerequisite result: {path}")
    return read_csv(path)


def main() -> None:
    physical = [
        row for row in require(output_path("overall_comparison") / "scene_details.csv")
        if row["method"] != "session_calibrated_als"
    ]
    classical = require(output_path("classical_baselines") / "scene_details.csv")
    em_sbl = require(output_path("em_sbl") / "scene_details.csv")
    controlled = require(output_path("observation_budget") / "support8_scene_details.csv")
    all_rows = physical + classical + em_sbl + controlled
    write_csv(DESTINATION / "E01_scene_details.csv", all_rows)
    write_csv(DESTINATION / "E01_overall_comparison.csv", summarize(all_rows, ("method",)))

    snr_methods = {"music", "cov_setnet_support8", "nominal_als", "sc_phdoa"}
    write_csv(
        DESTINATION / "E06_snr_robustness.csv",
        summarize([row for row in all_rows if row["method"] in snr_methods], ("method", "snr")),
    )
    write_csv(
        DESTINATION / "E07_source_count_breakdown.csv",
        summarize(all_rows, ("method", "K_true")),
    )
    write_csv(
        DESTINATION / "E09_array_condition_breakdown.csv",
        summarize(all_rows, ("method", "case")),
    )
    print(f"Assembled paper tables: {DESTINATION}")


if __name__ == "__main__":
    main()
