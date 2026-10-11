"""Guard-scene residual calibration and deterministic routing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np


@dataclass(frozen=True)
class RouteDecision:
    use_residual: bool
    score: float
    threshold: float

    @property
    def name(self) -> str:
        return "residual" if self.use_residual else "nominal"


class GuardRouter:
    """Apply the locked projection-score threshold at session level.

    The score is nominal projection error minus context-calibrated projection
    error on guard scenes. This is a branch-utility rule, not a calibrated
    estimate of hardware-drift magnitude.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)

    def decide(self, nominal_score: float, context_score: float) -> RouteDecision:
        score = float(nominal_score - context_score)
        return RouteDecision(score > self.threshold, score, self.threshold)


class ResidualCalibrationCorrector:
    """Add the learned guard-conditioned correction to selected state entries."""

    def __init__(self, estimator: Any, dimensions: np.ndarray) -> None:
        self.estimator = estimator
        self.dimensions = np.asarray(dimensions, np.int64)

    @classmethod
    def load(cls, path: Union[str, Path], dimensions: np.ndarray) -> "ResidualCalibrationCorrector":
        import joblib

        return cls(joblib.load(Path(path)), dimensions)

    def correct(self, features: np.ndarray, initial: np.ndarray) -> np.ndarray:
        features = np.atleast_2d(np.asarray(features, np.float32))
        initial = np.atleast_2d(np.asarray(initial, np.float64))
        estimate = initial.copy()
        correction = np.asarray(self.estimator.predict(features), np.float64)
        estimate[:, self.dimensions] += correction
        return np.clip(estimate, -1.0, 1.0).astype(np.float32)


def build_guard_summary(
    nominal_score: float,
    context_score: float,
    calibration_gates: np.ndarray,
    initial_calibration: np.ndarray,
) -> np.ndarray:
    initial = np.asarray(initial_calibration, np.float32)
    gates = np.asarray(calibration_gates, np.float32).reshape(-1)
    eta_norm = float(np.linalg.norm(initial) / np.sqrt(len(initial)))
    return np.asarray(
        [nominal_score, context_score, nominal_score - context_score, *gates, eta_norm],
        np.float32,
    )
