"""Frozen nominal-array covariance EM-SBL baseline."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize

from .classical import NominalArray, spherical_distance


def normalized_scene_covariance(snapshots: np.ndarray) -> np.ndarray:
    """Match the frozen V18.4 covariance cache used by the paper SBL run."""
    signal = np.asarray(snapshots, np.complex64)
    centered = signal - signal.mean(axis=1, keepdims=True)
    covariance = centered @ centered.conj().T / max(centered.shape[1], 1)
    power = max(float(np.trace(covariance).real / signal.shape[0]), 1e-10)
    normalized = covariance / power
    return (
        normalized.real.astype(np.float32).astype(np.complex64)
        + 1j * normalized.imag.astype(np.float32).astype(np.complex64)
    )


class CovarianceSBL:
    def __init__(self, array: NominalArray, device: str, iterations: int = 12, damping: float = 0.65, noise_floor: float = 0.002):
        self.array, self.device = array, device
        self.iterations, self.damping, self.noise_floor = iterations, damping, noise_floor
        self.grid = np.asarray(array.grid, np.float32)
        steering = np.asarray(array.grid_steering, np.complex64)
        self.steering = torch.as_tensor(steering, device=device)
        self.steering_h = self.steering.conj().transpose(0, 1)
        self.identity = torch.eye(steering.shape[0], dtype=torch.complex64, device=device)

    @torch.no_grad()
    def powers(self, covariance: np.ndarray, counts: np.ndarray) -> np.ndarray:
        covariance = torch.as_tensor(covariance, dtype=torch.complex64, device=self.device)
        covariance = 0.5 * (covariance + covariance.conj().transpose(-1, -2))
        sensors = covariance.shape[-1]
        beam = torch.einsum("mg,bmn,ng->bg", self.steering.conj(), covariance, self.steering).real / float(sensors * sensors)
        power = torch.clamp(beam, min=1e-7, max=1.0)
        eigenvalues = torch.linalg.eigvalsh(covariance).real
        noise = torch.stack([torch.clamp(eigenvalues[row, :max(sensors - int(count), 1)].mean(), min=self.noise_floor) for row, count in enumerate(counts.tolist())])
        for _ in range(self.iterations):
            evidence = (self.steering.unsqueeze(0) * power.unsqueeze(1)) @ self.steering_h.unsqueeze(0) + noise[:, None, None] * self.identity
            inverse = torch.linalg.inv(evidence)
            q = torch.sum(self.steering.conj().unsqueeze(0) * (inverse @ self.steering), dim=1).real.clamp_min(1e-8)
            middle = inverse @ covariance @ inverse
            t = torch.einsum("mg,bmn,ng->bg", self.steering.conj(), middle, self.steering).real.clamp_min(0.0)
            updated = torch.clamp(power - power.square() * q + power.square() * t, min=1e-8, max=2.0)
            power = (1.0 - self.damping) * power + self.damping * updated
        return power.cpu().numpy().astype(np.float32)

    def _select(self, power: np.ndarray, count: int) -> np.ndarray:
        selected: list[np.ndarray] = []
        for index in np.argsort(-np.asarray(power, np.float64)):
            candidate = self.grid[int(index)]
            if all(spherical_distance(candidate, previous) >= 4.0 for previous in selected):
                selected.append(candidate)
                if len(selected) == count:
                    break
        if len(selected) < count:
            selected.extend(
                self.grid[int(index)]
                for index in np.argsort(-np.asarray(power, np.float64))
                if not any(np.allclose(self.grid[int(index)], old) for old in selected)
            )
        return np.asarray(selected[:count], np.float32)

    def _refine(self, initial: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        count = len(initial)
        lower = np.maximum(initial - 2.5, np.asarray([-60.0, 5.0]))
        upper = np.minimum(initial + 2.5, np.asarray([60.0, 80.0]))
        covariance = 0.5 * (covariance + covariance.conj().T)
        identity = np.eye(covariance.shape[0], dtype=np.complex128)

        def objective(value: np.ndarray) -> float:
            steering = self.array.steering(value.reshape(count, 2)).astype(np.complex128)
            gram = steering.conj().T @ steering + 1e-4 * np.eye(count)
            try:
                projector = steering @ np.linalg.solve(gram, steering.conj().T)
            except np.linalg.LinAlgError:
                return 1e6
            return float(np.trace((identity - projector) @ covariance).real)

        result = minimize(objective, initial.ravel(), method="L-BFGS-B", bounds=list(zip(lower.ravel(), upper.ravel())), options={"maxiter": 25, "ftol": 1e-12})
        return np.asarray(result.x.reshape(count, 2) if np.all(np.isfinite(result.x)) else initial, np.float32)

    def estimate(self, covariance: np.ndarray, count: int) -> np.ndarray:
        power = self.powers(np.asarray(covariance)[None], np.asarray([count], np.int64))[0]
        return self._refine(self._select(power, count), covariance)

    def estimate_batch(self, covariance: np.ndarray, counts: np.ndarray) -> list[np.ndarray]:
        powers = self.powers(covariance, counts)
        return [
            self._refine(self._select(powers[row], int(count)), covariance[row])
            for row, count in enumerate(counts)
        ]
