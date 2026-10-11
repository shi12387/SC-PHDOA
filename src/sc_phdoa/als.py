"""Robust multi-start calibrated alternating least squares."""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_CONFIG, ModelConfig
from .features import normalize_columns
from .manifold import (
    COARSE_GRID,
    ManifoldFunction,
    angles_from_mixing,
    project_with_starts,
    spherical_distance,
)


def solve_symbols(
    mixing: np.ndarray, snapshots: np.ndarray, ridge: float
) -> np.ndarray:
    count = mixing.shape[1]
    gram = mixing.conj().T @ mixing
    scale = max(float(np.trace(gram).real) / count, 1e-10)
    return np.linalg.solve(
        gram + ridge * scale * np.eye(count), mixing.conj().T @ snapshots
    )


def residual_fraction(
    mixing: np.ndarray, snapshots: np.ndarray, symbols: np.ndarray
) -> float:
    residual = snapshots - mixing @ symbols
    return float(
        np.vdot(residual, residual).real
        / max(np.vdot(snapshots, snapshots).real, 1e-10)
    )


def _select_starts(
    observed: np.ndarray,
    manifold: ManifoldFunction,
    primary_angle: np.ndarray,
    grid_manifold: np.ndarray,
    top_count: int,
    config: ModelConfig,
) -> list[np.ndarray]:
    observed = np.asarray(observed, np.complex128).reshape(-1)
    norm = np.linalg.norm(observed)
    if not np.isfinite(norm) or norm < 1e-10:
        return [np.asarray(primary_angle, np.float64)]
    observed /= norm
    denominator = np.maximum(np.sum(np.abs(grid_manifold) ** 2, axis=0), 1e-12)
    correlation = np.abs(grid_manifold.conj().T @ observed) ** 2 / denominator
    starts = [np.asarray(primary_angle, np.float64)]
    for index in np.argsort(-correlation):
        candidate = COARSE_GRID[int(index)]
        if min(float(spherical_distance(candidate, selected)) for selected in starts) >= config.multistart_nms_deg:
            starts.append(candidate)
        if len(starts) >= top_count + 1:
            break
    return starts


def _multistart_project(
    mixing: np.ndarray,
    manifold: ManifoldFunction,
    grid_manifold: np.ndarray,
    primary_angles: np.ndarray,
    top_count: int,
    config: ModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = [
        _select_starts(
            mixing[:, source], manifold, primary_angles[source], grid_manifold, top_count, config
        )
        for source in range(mixing.shape[1])
    ]
    return project_with_starts(mixing, manifold, starts)


def calibrated_als(
    snapshots: np.ndarray,
    initial_mixing: np.ndarray,
    manifold: ManifoldFunction,
    grid_manifold: np.ndarray | None = None,
    config: ModelConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    snapshots = np.asarray(snapshots, np.complex128)
    initial_mixing = np.asarray(initial_mixing, np.complex128)
    count = initial_mixing.shape[1]
    grid_manifold = manifold(COARSE_GRID) if grid_manifold is None else grid_manifold
    direct_angles = angles_from_mixing(initial_mixing, manifold)
    mixing, angles, initial_objectives = _multistart_project(
        initial_mixing,
        manifold,
        grid_manifold,
        direct_angles,
        config.initial_multistart_count,
        config,
    )
    anchor = normalize_columns(mixing)
    symbols = solve_symbols(mixing, snapshots, config.als_ridge)
    initial_residual = residual_fraction(mixing, snapshots, symbols)
    current_residual = initial_residual
    accepted = rejected_residual = rejected_condition = 0
    largest_shift = 0.0
    trajectory = [initial_residual]
    for _ in range(config.als_iterations):
        symbols = solve_symbols(mixing, snapshots, config.als_ridge)
        gram = symbols @ symbols.conj().T
        scale = max(float(np.trace(gram).real) / count, 1e-10)
        updated = snapshots @ symbols.conj().T @ np.linalg.inv(
            gram + config.als_ridge * scale * np.eye(count)
        )
        for source in range(count):
            phase = np.vdot(anchor[:, source], updated[:, source])
            if abs(phase) > 1e-10:
                updated[:, source] *= np.conj(phase) / abs(phase)
        blended = (1.0 - config.als_blend) * mixing + config.als_blend * updated
        candidate, candidate_angles, _ = _multistart_project(
            blended,
            manifold,
            grid_manifold,
            angles,
            config.iterative_multistart_count,
            config,
        )
        candidate_symbols = solve_symbols(candidate, snapshots, config.als_ridge)
        candidate_residual = residual_fraction(candidate, snapshots, candidate_symbols)
        condition = float(np.linalg.cond(candidate.conj().T @ candidate))
        shift = float(np.max(spherical_distance(candidate_angles, angles)))
        largest_shift = max(largest_shift, shift)
        residual_ok = np.isfinite(candidate_residual) and candidate_residual <= current_residual * (
            1.0 + config.residual_relative_tolerance
        )
        condition_ok = np.isfinite(condition) and condition <= config.maximum_steering_condition
        if residual_ok and condition_ok:
            mixing, angles, current_residual = candidate, candidate_angles, candidate_residual
            trajectory.append(current_residual)
            accepted += 1
        else:
            rejected_residual += int(not residual_ok)
            rejected_condition += int(not condition_ok)
            break
    diagnostics = {
        "initial_objective_mean": float(np.mean(initial_objectives)),
        "residual_initial": initial_residual,
        "residual_final": current_residual,
        "residual_gain": initial_residual - current_residual,
        "accepted_steps": accepted,
        "rejected_residual": rejected_residual,
        "rejected_condition": rejected_condition,
        "largest_shift": largest_shift,
        "condition": float(np.linalg.cond(mixing.conj().T @ mixing)),
        "trajectory": trajectory,
    }
    return mixing, angles, diagnostics
