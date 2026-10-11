"""Locked nominal-array MUSIC, MDL, and L-array ESPRIT baselines."""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.optimize import minimize

X_ARM = np.asarray([0, 1, 2, 3], np.int64)
Z_ARM = np.asarray([0, 4, 5, 6], np.int64)


def hermitian(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.complex128)
    return 0.5 * (value + value.conj().T)


def sample_covariance(snapshots: np.ndarray) -> np.ndarray:
    snapshots = np.asarray(snapshots, np.complex128)
    centered = snapshots - np.mean(snapshots, axis=1, keepdims=True)
    return hermitian(centered @ centered.conj().T / max(centered.shape[1], 1))


def source_count_mdl(covariance: np.ndarray, snapshots: int, max_k: int = 3) -> int:
    """Wax-Kailath MDL constrained to the paper's source support 1..3."""
    eigenvalues = np.maximum(np.linalg.eigvalsh(hermitian(covariance))[::-1].real, 1e-12)
    sensors = len(eigenvalues)
    candidates = list(range(1, min(max_k, sensors - 1) + 1))
    scores = []
    for count in candidates:
        noise = eigenvalues[count:]
        arithmetic = max(float(np.mean(noise)), 1e-12)
        geometric = float(np.exp(np.mean(np.log(noise))))
        likelihood = -snapshots * (sensors - count) * math.log(max(geometric / arithmetic, 1e-12))
        penalty = 0.5 * count * (2 * sensors - count) * math.log(max(snapshots, 2))
        scores.append(likelihood + penalty)
    return int(candidates[int(np.argmin(scores))])


def direction(angles: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles, np.float64)
    azimuth = np.deg2rad(angles[..., 0])
    elevation = np.deg2rad(angles[..., 1])
    return np.stack([
        np.cos(elevation) * np.sin(azimuth),
        np.cos(elevation) * np.cos(azimuth),
        np.sin(elevation),
    ], axis=-1)


def spherical_distance(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float(np.clip(direction(np.asarray(first)[None])[0] @ direction(np.asarray(second)[None])[0], -1.0, 1.0))
    return math.degrees(math.acos(cosine))


class NominalArray:
    def __init__(self, coordinates: np.ndarray, wavelength: float, grid_step_deg: float = 1.0):
        self.coordinates = np.asarray(coordinates, np.float64)
        self.wavelength = float(wavelength)
        azimuth = np.arange(-60.0, 60.0 + 0.5 * grid_step_deg, grid_step_deg)
        elevation = np.arange(5.0, 80.0 + 0.5 * grid_step_deg, grid_step_deg)
        azimuth_grid, elevation_grid = np.meshgrid(azimuth, elevation, indexing="xy")
        self.grid = np.stack([azimuth_grid.ravel(), elevation_grid.ravel()], axis=1)
        self.grid_steering = self.steering(self.grid).astype(np.complex64)

    def steering(self, angles: np.ndarray) -> np.ndarray:
        angles = np.asarray(angles, np.float64)
        azimuth = np.deg2rad(angles[..., 0])
        elevation = np.deg2rad(angles[..., 1])
        ux = np.cos(elevation) * np.sin(azimuth)
        uz = np.sin(elevation)
        phase = self.coordinates[0, :, None] * ux.reshape(1, -1) + self.coordinates[2, :, None] * uz.reshape(1, -1)
        return np.exp(-2j * np.pi * phase / self.wavelength)


def noise_basis(covariance: np.ndarray, count: int) -> np.ndarray:
    _, vectors = np.linalg.eigh(hermitian(covariance))
    return vectors[:, :max(vectors.shape[1] - int(count), 1)]


def _refine_music(center: np.ndarray, basis: np.ndarray, array: NominalArray, radius: float, iterations: int) -> np.ndarray:
    center = np.asarray(center, np.float64)
    lower = np.maximum(center - radius, np.asarray([-60.0, 5.0]))
    upper = np.minimum(center + radius, np.asarray([60.0, 80.0]))

    def objective(value: np.ndarray) -> float:
        steering = array.steering(np.asarray(value)[None])[:, 0]
        projected = basis.conj().T @ steering
        return float(np.vdot(projected, projected).real / max(np.vdot(steering, steering).real, 1e-12))

    result = minimize(objective, np.clip(center, lower, upper), method="L-BFGS-B", bounds=list(zip(lower, upper)), options={"maxiter": iterations, "ftol": 1e-12})
    return np.asarray(result.x if np.all(np.isfinite(result.x)) else center, np.float32)


def music_2d(covariance: np.ndarray, count: int, array: NominalArray, nms_deg: float = 4.0, local_radius_deg: float = 1.5, local_iterations: int = 30) -> np.ndarray:
    basis = noise_basis(covariance, count)
    denominator = np.sum(np.abs(basis.conj().T @ array.grid_steering) ** 2, axis=0)
    selected: list[np.ndarray] = []
    for index in np.argsort(denominator):
        candidate = array.grid[int(index)]
        if all(spherical_distance(candidate, previous) >= nms_deg for previous in selected):
            selected.append(candidate)
            if len(selected) == int(count):
                break
    if len(selected) != int(count):
        raise RuntimeError(f"MUSIC selected {len(selected)}/{count} peaks")
    return np.stack([_refine_music(value, basis, array, local_radius_deg, local_iterations) for value in selected])


def _arm_esprit(covariance: np.ndarray, count: int) -> np.ndarray:
    _, vectors = np.linalg.eigh(hermitian(covariance))
    signal = vectors[:, -int(count):]
    rotation = np.linalg.pinv(signal[:-1]) @ signal[1:]
    return np.clip(np.real(-np.angle(np.linalg.eigvals(rotation)) / np.pi), -0.999, 0.999)


def l_shaped_esprit(covariance: np.ndarray, count: int, array: NominalArray) -> np.ndarray:
    count = int(count)
    ux_values = _arm_esprit(covariance[np.ix_(X_ARM, X_ARM)], count)
    uz_values = _arm_esprit(covariance[np.ix_(Z_ARM, Z_ARM)], count)
    _, vectors = np.linalg.eigh(hermitian(covariance))
    signal = vectors[:, -count:]
    projector = signal @ signal.conj().T
    best_angles, best_score = None, float("inf")
    for permutation in itertools.permutations(range(count)):
        uz = uz_values[np.asarray(permutation)]
        elevation = np.arcsin(np.clip(uz, math.sin(math.radians(5.0)), math.sin(math.radians(80.0))))
        azimuth = np.arcsin(np.clip(ux_values / np.maximum(np.cos(elevation), 0.15), -math.sin(math.radians(60.0)), math.sin(math.radians(60.0))))
        angles = np.stack([np.rad2deg(azimuth), np.rad2deg(elevation)], axis=1)
        steering = array.steering(angles)
        basis, _ = np.linalg.qr(steering, mode="reduced")
        candidate = basis[:, :count] @ basis[:, :count].conj().T
        score = float(np.linalg.norm(projector - candidate, "fro"))
        if score < best_score:
            best_angles, best_score = angles, score
    if best_angles is None or not np.all(np.isfinite(best_angles)):
        raise RuntimeError("ESPRIT pairing failed")
    return np.asarray(best_angles, np.float32)
