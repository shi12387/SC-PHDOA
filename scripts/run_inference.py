"""Run the final SC-PHDOA pipeline on selected external sessions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.inference import SCPHDOAInferencePipeline
from sc_phdoa.protocol import dataset_path, load_protocol, output_path


# Edit these defaults for direct execution from an IDE.
PROTOCOL = load_protocol()
EXTERNAL_MAT = dataset_path("external_test")
OUTPUT_DIRECTORY = output_path("inference")
STATE_INDICES = (0, 1, 2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final SC-PHDOA inference")
    parser.add_argument("--mat", type=Path, default=EXTERNAL_MAT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--states", default=",".join(map(str, STATE_INDICES)))
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    states = [int(value) for value in arguments.states.split(",") if value.strip()]
    arguments.output.mkdir(parents=True, exist_ok=True)
    seed = int(PROTOCOL["random_seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pipeline = SCPHDOAInferencePipeline.from_files(arguments.mat, device=arguments.device)
    summaries = []
    started = time.perf_counter()
    print("SC-PHDOA: ALS=1, ridge=0.0003, blend=0.65, final IQ refiner disabled")
    for position, state in enumerate(states, 1):
        print(f"[{position}/{len(states)}] session {state}")
        prediction = pipeline.predict_session(state)
        prediction.save(arguments.output / f"session_{state:04d}")
        summaries.append(prediction.summary())
        print(f"  route={prediction.calibration_route}")
    summary = {
        "external_mat": str(arguments.mat.resolve()),
        "device": arguments.device,
        "states": states,
        "als_iterations": int(PROTOCOL["als_iterations"]),
        "final_iq_refiner": bool(PROTOCOL["final_iq_refiner"]),
        "elapsed_seconds": time.perf_counter() - started,
        "predictions": summaries,
    }
    with (arguments.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved: {arguments.output}")


if __name__ == "__main__":
    main()
