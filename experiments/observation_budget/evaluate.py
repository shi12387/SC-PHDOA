"""E02: controlled 8-versus-12-support observation-budget comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.baselines.setnet_evaluation import evaluate_setnet_pair
from sc_phdoa.evaluation import paired_session_bootstrap, read_csv, summarize, write_csv
from sc_phdoa.protocol import dataset_path, output_path

MAT_PATH = dataset_path("external_test")
OUTPUT_DIRECTORY = output_path("observation_budget")
CHECKPOINT_ROOT = ROOT / "checkpoints" / "controlled_setnets"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    combined = []
    for support_count in (8, 12):
        destination = OUTPUT_DIRECTORY / f"support{support_count}_scene_details.csv"
        existing = read_csv(destination) if destination.is_file() else []
        rows = evaluate_setnet_pair(
            MAT_PATH,
            CHECKPOINT_ROOT / f"support{support_count}",
            support_count,
            DEVICE,
            existing,
        )
        write_csv(destination, rows)
        combined.extend(rows)
    write_csv(OUTPUT_DIRECTORY / "scene_details.csv", combined)
    write_csv(OUTPUT_DIRECTORY / "method_summary.csv", summarize(combined, ("method",)))
    comparisons = []
    for representation in ("cov", "iq"):
        comparisons.extend(paired_session_bootstrap(
            combined,
            f"{representation}_setnet_support8",
            f"{representation}_setnet_support12",
        ))
    write_csv(OUTPUT_DIRECTORY / "support12_minus_support8_bootstrap.csv", comparisons)


if __name__ == "__main__":
    main()
