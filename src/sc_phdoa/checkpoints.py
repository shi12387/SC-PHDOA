"""Frozen artifact discovery and loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .calibration import SessionCalibrationNetwork
from .config import DEFAULT_CONFIG
from .guard_routing import GuardRouter, ResidualCalibrationCorrector
from .source_count import SourceCountEstimator


@dataclass(frozen=True)
class CheckpointPaths:
    source_count: Path
    session_calibration: Path
    guard_residual: Path
    guard_threshold: Path

    @classmethod
    def bundled(cls) -> "CheckpointPaths":
        root = Path(__file__).resolve().parents[2] / "checkpoints"
        return cls(
            source_count=root / "source_count" / "estimator.joblib",
            session_calibration=root / "session_calibration" / "network.pt",
            guard_residual=root / "guard_routing" / "residual_model.joblib",
            guard_threshold=root / "guard_routing" / "threshold.json",
        )

    def validate(self) -> None:
        missing = [str(path) for path in self.__dict__.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing frozen artifacts:\n  " + "\n  ".join(missing))


@dataclass
class LoadedComponents:
    source_count: SourceCountEstimator
    session_calibrator: SessionCalibrationNetwork
    guard_corrector: ResidualCalibrationCorrector
    guard_router: GuardRouter


def load_components(
    paths: CheckpointPaths | None = None, device: str = "cpu"
) -> LoadedComponents:
    paths = paths or CheckpointPaths.bundled()
    paths.validate()
    source_count = SourceCountEstimator.load(paths.source_count)
    checkpoint = torch.load(
        paths.session_calibration, map_location=device, weights_only=True
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"unexpected calibration checkpoint: {paths.session_calibration}")
    calibration = SessionCalibrationNetwork().to(device)
    calibration.load_state_dict(checkpoint["model"], strict=True)
    calibration.eval()
    for parameter in calibration.parameters():
        parameter.requires_grad_(False)
    corrector = ResidualCalibrationCorrector.load(
        paths.guard_residual,
        np.arange(DEFAULT_CONFIG.calibration_dim, dtype=np.int64),
    )
    threshold_payload = json.loads(paths.guard_threshold.read_text(encoding="utf-8"))
    if "threshold" not in threshold_payload:
        raise ValueError(f"missing threshold value: {paths.guard_threshold}")
    threshold = float(threshold_payload["threshold"])
    return LoadedComponents(
        source_count=source_count,
        session_calibrator=calibration,
        guard_corrector=corrector,
        guard_router=GuardRouter(threshold),
    )
