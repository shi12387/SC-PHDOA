"""Train analytic-only, IQ-only, and full session-calibration variants."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.protocol import dataset_path, output_path

TRAINER = ROOT / "scripts" / "train_session_calibration.py"
MAT_PATH = dataset_path("session_calibration_training")
CACHE_DIRECTORY = output_path("session_calibration_cache")
OUTPUT_DIRECTORY = output_path("calibration_input_ablation")


def main() -> None:
    for variant in ("analytic_only", "iq_only", "full"):
        command = [
            sys.executable,
            str(TRAINER),
            "--mat", str(MAT_PATH),
            "--cache", str(CACHE_DIRECTORY),
            "--output", str(OUTPUT_DIRECTORY / variant),
            "--variant", variant,
        ]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
