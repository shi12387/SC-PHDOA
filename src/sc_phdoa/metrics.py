"""Set-valued angular metrics used by all paper experiments."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .manifold import direction


def angular_cost(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    cosine = np.clip(direction(prediction) @ direction(target).T, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def matched_errors(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    cost = angular_cost(np.asarray(prediction), np.asarray(target))
    rows, columns = linear_sum_assignment(cost)
    return cost[rows, columns]


def ospa(
    prediction: np.ndarray,
    target: np.ndarray,
    cutoff: float = 20.0,
    order: float = 2.0,
) -> float:
    prediction = np.asarray(prediction, np.float64).reshape(-1, 2)
    target = np.asarray(target, np.float64).reshape(-1, 2)
    if len(prediction) == 0 or len(target) == 0:
        return cutoff
    cost = np.minimum(angular_cost(prediction, target), cutoff) ** order
    rows, columns = linear_sum_assignment(cost)
    total = float(np.sum(cost[rows, columns]))
    total += abs(len(prediction) - len(target)) * cutoff**order
    return float((total / max(len(prediction), len(target))) ** (1.0 / order))


def manifold_error_and_coherence(
    estimate: np.ndarray, oracle: np.ndarray
) -> tuple[float, float]:
    """Return scale-invariant manifold error and normalized coherence."""
    estimate = np.asarray(estimate, np.complex128)
    oracle = np.asarray(oracle, np.complex128)
    denominator = np.sum(np.abs(estimate) ** 2, axis=0) + 1e-12
    scale = np.sum(np.conj(estimate) * oracle, axis=0) / denominator
    residual = oracle - estimate * scale[None]
    error = np.linalg.norm(residual, axis=0) / (
        np.linalg.norm(oracle, axis=0) + 1e-12
    )
    coherence = np.abs(np.sum(np.conj(estimate) * oracle, axis=0)) / (
        np.linalg.norm(estimate, axis=0)
        * np.linalg.norm(oracle, axis=0)
        + 1e-12
    )
    return float(np.mean(error)), float(np.mean(coherence))


def calibration_component_errors(
    estimate: tuple[np.ndarray, np.ndarray, np.ndarray],
    oracle: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacing: float,
    coupling_pairs: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> tuple[float, float, float]:
    """Return gain EVM, position RMSE/d, and complex coupling RMSE."""
    gain_hat, coupling_hat, position_hat = estimate
    gain, coupling, position = oracle
    gain_error = float(
        np.sqrt(np.mean(np.abs(gain_hat[1:] - gain[1:]) ** 2))
        / max(np.sqrt(np.mean(np.abs(gain[1:]) ** 2)), 1e-12)
    )
    position_error = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (position_hat[:, 1:] - position[:, 1:]) ** 2,
                    axis=0,
                )
            )
        )
        / float(spacing)
    )
    pair_error = np.asarray(
        [
            coupling_hat[left, right] - coupling[left, right]
            for left, right in coupling_pairs
        ]
    )
    coupling_error = float(np.sqrt(np.mean(np.abs(pair_error) ** 2)))
    return gain_error, position_error, coupling_error
