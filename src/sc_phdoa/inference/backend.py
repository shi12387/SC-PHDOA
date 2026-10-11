"""Deterministic signal-processing backend for final inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..als import calibrated_als
from ..config import DEFAULT_CONFIG
from ..dataset import SessionDataset
from ..features import aligned_iq_block, analytic_source_features, normalize_columns
from ..guard_features import build_guard_features, projection_score, source_residual_token
from ..manifold import COARSE_GRID, angles_from_mixing, decode_calibration, make_manifold, nominal_manifold
from ..synchronization import estimate_synchronization_and_mixing

DIAGNOSTIC_KEYS = (
    "residual_final", "condition", "accepted_steps", "residual_gain", "largest_shift"
)


@dataclass
class SignalProcessingBackend:
    dataset: SessionDataset

    @classmethod
    def load(
        cls, external_mat: Path, validate_identity: bool = True
    ) -> "SignalProcessingBackend":
        return cls(SessionDataset.load_external(external_mat, validate_identity))

    @property
    def state_count(self) -> int:
        return len(self.dataset.state_ids)

    @property
    def payload_start(self) -> int:
        return DEFAULT_CONFIG.payload_start

    @property
    def preamble(self) -> np.ndarray:
        return self.dataset.preamble

    @property
    def maximum_delay(self) -> int:
        return self.dataset.maximum_delay

    def read_session(self, state_index: int) -> list[np.ndarray]:
        return self.dataset.read_session(state_index)

    @staticmethod
    def estimate_snr(snapshots: np.ndarray, source_count: int) -> float:
        centered = snapshots - snapshots.mean(axis=1, keepdims=True)
        covariance = centered @ centered.conj().T / max(centered.shape[1], 1)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance).real, 1e-12)
        noise = float(np.mean(eigenvalues[: max(len(eigenvalues) - source_count, 1)]))
        signal = max(float(eigenvalues.sum() - len(eigenvalues) * noise), 1e-12)
        ratio = signal / (len(eigenvalues) * noise)
        return float(np.clip(10.0 * np.log10(max(ratio, 1e-12)), -15.0, 30.0))

    def build_source_frontend(
        self, snapshots: np.ndarray, source_count: int, snr_db: float
    ) -> dict[str, Any]:
        mixing, synchronization = estimate_synchronization_and_mixing(
            snapshots, self.preamble, self.maximum_delay, source_count
        )
        aligned = np.stack([
            aligned_iq_block(
                snapshots, self.preamble, synchronization[source, 0],
                synchronization[source, 1], DEFAULT_CONFIG.preamble_samples,
            )
            for source in range(source_count)
        ])
        analytic = analytic_source_features(
            mixing, synchronization, source_count, snr_db, self.maximum_delay
        )[:source_count]
        return {
            "count": source_count,
            "mixing": normalize_columns(mixing),
            "synchronization": synchronization,
            "analytic": analytic,
            "aligned_iq": aligned.astype(np.float32),
            "snr_db": float(snr_db),
        }

    def _decode_manifold(self, calibration_vector: np.ndarray):
        calibration = decode_calibration(
            calibration_vector,
            self.dataset.nominal_coordinates,
            self.dataset.spacing,
            self.dataset.coupling_pairs,
        )
        return make_manifold(calibration, self.dataset.wavelength)

    def build_session_manifold(
        self, calibration_vector: np.ndarray
    ) -> dict[str, Any]:
        selected = self._decode_manifold(calibration_vector)
        return {
            "manifold": selected,
            "grid": selected(COARSE_GRID),
            "mode": "session",
            "use_residual": False,
            "guard_features": np.full(7, np.nan, np.float32),
            "initial_calibration": np.asarray(calibration_vector, np.float32),
            "corrected_calibration": np.asarray(calibration_vector, np.float32),
        }

    def build_nominal_manifold(self) -> dict[str, Any]:
        selected = nominal_manifold(
            self.dataset.nominal_coordinates, self.dataset.wavelength
        )
        return {
            "manifold": selected,
            "grid": selected(COARSE_GRID),
            "mode": "nominal",
            "use_residual": False,
            "guard_features": np.full(7, np.nan, np.float32),
            "initial_calibration": np.zeros(50, np.float32),
            "corrected_calibration": np.zeros(50, np.float32),
        }

    def build_routed_manifold(
        self,
        state_index: int,
        frontends: list[dict[str, Any]],
        calibration: dict[str, np.ndarray],
        residual_model: Any,
        residual_dimensions: np.ndarray,
        route_threshold: float,
    ) -> dict[str, Any]:
        nominal = nominal_manifold(
            self.dataset.nominal_coordinates, self.dataset.wavelength
        )
        initial = np.asarray(calibration["final"], np.float32)
        gates = np.asarray(calibration["gates"], np.float32)
        context = self._decode_manifold(initial)
        calibration_norm = float(np.linalg.norm(initial) / np.sqrt(len(initial)))
        nominal_scores: list[float] = []
        context_scores: list[float] = []
        tokens: list[np.ndarray] = []
        for scene in self.dataset.guard_scenes[state_index]:
            frontend = frontends[int(scene)]
            mixing = frontend["mixing"]
            source_count = frontend["count"]
            nominal_scores.append(projection_score(mixing, nominal))
            context_scores.append(projection_score(mixing, context))
            try:
                angles = angles_from_mixing(mixing, context)
                predicted = context(angles)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                angles = np.tile([0.0, 42.5], (source_count, 1))
                predicted = context(angles)
            for source in range(source_count):
                tokens.append(source_residual_token(
                    mixing[:, source], predicted[:, source], angles[source], source_count,
                    frontend["snr_db"], gates, calibration_norm,
                ))
        nominal_score = float(np.mean(nominal_scores))
        context_score = float(np.mean(context_scores))
        guard_summary, residual_features = build_guard_features(
            initial, gates, nominal_score, context_score, tokens
        )
        corrected = initial[None].astype(np.float64)
        correction = np.asarray(residual_model.predict(residual_features[None]), np.float64)
        corrected[:, residual_dimensions] += correction
        corrected = np.clip(corrected, -1.0, 1.0).astype(np.float32)[0]
        use_residual = bool(guard_summary[2] > route_threshold)
        selected = self._decode_manifold(corrected) if use_residual else nominal
        return {
            "manifold": selected,
            "grid": selected(COARSE_GRID),
            "mode": "residual" if use_residual else "nominal",
            "use_residual": use_residual,
            "guard_features": guard_summary,
            "initial_calibration": initial,
            "corrected_calibration": corrected,
            "gates": gates,
        }

    def estimate_target_doas(
        self, snapshots: np.ndarray, frontend: dict[str, Any], routed_manifold: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            _, angles, diagnostics = calibrated_als(
                snapshots, frontend["mixing"], routed_manifold["manifold"],
                routed_manifold["grid"],
            )
            diagnostic = np.asarray(
                [float(diagnostics.get(key, 0.0)) for key in DIAGNOSTIC_KEYS], np.float32
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            angles = angles_from_mixing(frontend["mixing"], routed_manifold["manifold"])
            diagnostic = np.asarray([1.0, 1e4, 0.0, 0.0, 180.0], np.float32)
        return np.asarray(angles, np.float32), diagnostic
