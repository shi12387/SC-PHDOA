"""Validate datasets, checkpoints, dependencies, and the final protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
import scipy
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.checkpoints import CheckpointPaths, load_components
from sc_phdoa.protocol import (
    PROTOCOL_PATH,
    dataset_path,
    load_protocol,
    output_path,
    protocol_fingerprint,
)

EVALUATION_KEYS = {
    "IQ_tensors", "num_sources_all", "doas_all", "array_state_id_all",
    "scene_within_state_all", "evaluation_role_all", "nominal_array_coords",
    "shared_preamble_real", "shared_preamble_imag",
}
DATASET_REQUIRED_KEYS = {
    "source_count_training": {
        "IQ_tensors", "num_sources_all", "geometry_id_all",
        "shared_preamble_real", "shared_preamble_imag",
    },
    "session_calibration_training": {
        "IQ_tensors", "num_sources_all", "doas_all", "array_state_id_all",
        "scene_within_state_all", "nominal_array_coords",
        "sensor_gain_real", "sensor_gain_imag", "coupling_real",
        "coupling_imag", "actual_array_coords_all",
        "shared_preamble_real", "shared_preamble_imag",
    },
}


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate reproduction inputs")
    parser.add_argument(
        "--hash-datasets", action="store_true",
        help="Compute full SHA-256 hashes for MAT files (slow for large files).",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    protocol = load_protocol()
    errors: list[str] = []
    datasets = []
    for name in protocol["datasets"]:
        path = dataset_path(name)
        record = {"name": name, "path": str(path), "exists": path.is_file()}
        if not path.is_file():
            errors.append(f"missing dataset: {path}")
        else:
            record["bytes"] = path.stat().st_size
            try:
                with h5py.File(path, "r") as handle:
                    required_keys = DATASET_REQUIRED_KEYS.get(
                        name, EVALUATION_KEYS
                    )
                    missing_keys = sorted(required_keys.difference(handle.keys()))
                    record["missing_required_keys"] = missing_keys
                    if missing_keys:
                        errors.append(f"{path.name}: missing keys {missing_keys}")
            except OSError as error:
                record["hdf5_error"] = str(error)
                errors.append(f"cannot open {path}: {error}")
            if arguments.hash_datasets:
                record["sha256"] = sha256(path)
        datasets.append(record)

    checkpoint_paths = CheckpointPaths.bundled()
    checkpoints = []
    for name, path in (
        ("source_count", checkpoint_paths.source_count),
        ("session_calibration", checkpoint_paths.session_calibration),
        ("guard_routing", checkpoint_paths.guard_residual),
        ("guard_threshold", checkpoint_paths.guard_threshold),
    ):
        record = {"name": name, "path": str(path), "exists": path.is_file()}
        if path.is_file():
            record["bytes"] = path.stat().st_size
            record["sha256"] = sha256(path)
        else:
            errors.append(f"missing checkpoint: {path}")
        checkpoints.append(record)

    experiment_manifest_path = ROOT / "checkpoints" / "EXPERIMENT_CHECKPOINTS.json"
    if not experiment_manifest_path.is_file():
        errors.append(f"missing checkpoint manifest: {experiment_manifest_path}")
    else:
        experiment_manifest = json.loads(experiment_manifest_path.read_text(encoding="utf-8"))
        for relative, expected_hash in experiment_manifest["files"].items():
            path = ROOT / "checkpoints" / relative
            record = {"name": relative, "path": str(path), "exists": path.is_file()}
            if not path.is_file():
                errors.append(f"missing checkpoint: {path}")
            else:
                record["bytes"] = path.stat().st_size
                record["sha256"] = sha256(path)
                if record["sha256"].upper() != expected_hash.upper():
                    errors.append(f"checkpoint hash mismatch: {path}")
            checkpoints.append(record)

    loaded_parameters = None
    try:
        components = load_components(checkpoint_paths, device="cpu")
        loaded_parameters = sum(
            parameter.numel()
            for parameter in components.session_calibrator.parameters()
        )
    except Exception as error:
        errors.append(f"checkpoint loading failed: {type(error).__name__}: {error}")

    report = {
        "ok": not errors,
        "errors": errors,
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_fingerprint": protocol_fingerprint(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
        "joblib": joblib.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "datasets": datasets,
        "checkpoints": checkpoints,
        "session_calibration_parameters": loaded_parameters,
    }
    destination = output_path("reproduction") / "environment_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
