"""L-array manifold construction and calibration-vector decoding."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares

from .config import DEFAULT_CONFIG

ManifoldFunction = Callable[[np.ndarray], np.ndarray]


def direction(angles: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles, np.float64)
    azimuth = np.deg2rad(angles[..., 0])
    elevation = np.deg2rad(angles[..., 1])
    return np.stack(
        [
            np.cos(elevation) * np.sin(azimuth),
            np.cos(elevation) * np.cos(azimuth),
            np.sin(elevation),
        ],
        axis=-1,
    )


def spherical_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    cosine = np.sum(direction(left) * direction(right), axis=-1)
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def steering_from_coordinates(
    coordinates: np.ndarray, wavelength: float, angles: np.ndarray
) -> np.ndarray:
    vectors = direction(angles)
    phase = (
        coordinates[0, :, None] * vectors[..., 0].reshape(1, -1)
        + coordinates[2, :, None] * vectors[..., 2].reshape(1, -1)
    )
    return np.exp(-1j * 2.0 * np.pi * phase / wavelength)


def effective_manifold(
    angles: np.ndarray,
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray],
    wavelength: float,
) -> np.ndarray:
    gain, coupling, coordinates = calibration
    steering = steering_from_coordinates(coordinates, wavelength, angles)
    return coupling @ (gain[:, None] * steering)


def make_manifold(
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray], wavelength: float
) -> ManifoldFunction:
    return lambda angles: effective_manifold(angles, calibration, wavelength)


def nominal_manifold(coordinates: np.ndarray, wavelength: float) -> ManifoldFunction:
    sensors = coordinates.shape[1]
    calibration = (
        np.ones(sensors, np.complex128),
        np.eye(sensors, dtype=np.complex128),
        np.asarray(coordinates, np.float64),
    )
    return make_manifold(calibration, wavelength)


def decode_calibration(
    values: np.ndarray,
    nominal_coordinates: np.ndarray,
    spacing: float,
    coupling_pairs: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, np.float64)
    cursor = 0
    gain = np.ones(7, np.complex128)
    real = values[cursor : cursor + 6] * DEFAULT_CONFIG.gain_scale
    cursor += 6
    imag = values[cursor : cursor + 6] * DEFAULT_CONFIG.gain_scale
    cursor += 6
    gain[1:] += real + 1j * imag
    coordinates = np.asarray(nominal_coordinates, np.float64).copy()
    position_scale = DEFAULT_CONFIG.position_scale_spacing * spacing
    coordinates[0, 1:] += values[cursor : cursor + 6] * position_scale
    cursor += 6
    coordinates[2, 1:] += values[cursor : cursor + 6] * position_scale
    cursor += 6
    pair_count = len(coupling_pairs)
    real = values[cursor : cursor + pair_count] * DEFAULT_CONFIG.coupling_scale
    cursor += pair_count
    imag = values[cursor : cursor + pair_count] * DEFAULT_CONFIG.coupling_scale
    coupling = np.eye(7, dtype=np.complex128)
    for index, (first, second) in enumerate(coupling_pairs):
        coupling[first, second] = real[index] + 1j * imag[index]
        coupling[second, first] = coupling[first, second]
    return gain, coupling, coordinates


def _nominal_analytic_initial(observed: np.ndarray) -> np.ndarray:
    x_arm = observed[[0, 1, 2, 3]]
    z_arm = observed[[0, 4, 5, 6]]
    ux = np.clip(
        -np.angle(np.sum(np.conj(x_arm[:-1]) * x_arm[1:])) / np.pi,
        -np.sin(np.deg2rad(60.0)),
        np.sin(np.deg2rad(60.0)),
    )
    uz = np.clip(
        -np.angle(np.sum(np.conj(z_arm[:-1]) * z_arm[1:])) / np.pi,
        np.sin(np.deg2rad(5.0)),
        np.sin(np.deg2rad(80.0)),
    )
    elevation = np.arcsin(float(uz))
    azimuth = np.arcsin(
        float(
            np.clip(
                ux / max(np.cos(elevation), 0.18),
                -np.sin(np.deg2rad(60.0)),
                np.sin(np.deg2rad(60.0)),
            )
        )
    )
    return np.rad2deg([azimuth, elevation])


def _normalized_nominal_steering(angles: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles, np.float64)
    vectors = direction(angles)
    x_position = np.asarray([0, 1, 2, 3, 0, 0, 0], np.float64)
    z_position = np.asarray([0, 0, 0, 0, 1, 2, 3], np.float64)
    phase = (
        x_position[:, None] * vectors[..., 0].reshape(1, -1)
        + z_position[:, None] * vectors[..., 2].reshape(1, -1)
    )
    return np.exp(-1j * np.pi * phase)


def nominal_initial_angle(observed: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed, np.complex128).reshape(-1)
    norm = np.linalg.norm(observed)
    if not np.isfinite(norm) or norm < 1e-10:
        return np.asarray([0.0, 42.5], np.float32)
    observed = observed / norm
    initial = _nominal_analytic_initial(observed)

    def residual(angle: np.ndarray) -> np.ndarray:
        column = _normalized_nominal_steering(np.asarray(angle)[None])[:, 0]
        denominator = max(float(np.vdot(column, column).real), 1e-12)
        coefficient = np.vdot(column, observed) / denominator
        error = observed - coefficient * column
        return np.concatenate([error.real, error.imag])

    try:
        result = least_squares(
            residual,
            initial,
            bounds=(
                [DEFAULT_CONFIG.azimuth_bounds_deg[0], DEFAULT_CONFIG.elevation_bounds_deg[0]],
                [DEFAULT_CONFIG.azimuth_bounds_deg[1], DEFAULT_CONFIG.elevation_bounds_deg[1]],
            ),
            method="trf",
            max_nfev=80,
        )
        return result.x.astype(np.float32)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return np.asarray(initial, np.float32)


def _column_residual(
    observed: np.ndarray, angle: np.ndarray, manifold: ManifoldFunction
) -> np.ndarray:
    column = manifold(np.asarray(angle)[None])[:, 0]
    denominator = max(float(np.vdot(column, column).real), 1e-12)
    coefficient = np.vdot(column, observed) / denominator
    error = observed - coefficient * column
    return np.concatenate([error.real, error.imag])


def fit_from_starts(
    observed: np.ndarray, manifold: ManifoldFunction, starts: Sequence[np.ndarray]
) -> tuple[np.ndarray, float]:
    observed = np.asarray(observed, np.complex128).reshape(-1)
    norm = np.linalg.norm(observed)
    if not np.isfinite(norm) or norm < 1e-10:
        return np.asarray([0.0, 42.5], np.float32), float("inf")
    observed /= norm
    lower = [DEFAULT_CONFIG.azimuth_bounds_deg[0], DEFAULT_CONFIG.elevation_bounds_deg[0]]
    upper = [DEFAULT_CONFIG.azimuth_bounds_deg[1], DEFAULT_CONFIG.elevation_bounds_deg[1]]
    best_angle, best_cost = None, float("inf")
    for start in starts:
        start = np.clip(np.asarray(start, np.float64).reshape(2), lower, upper)
        try:
            result = least_squares(
                lambda angle: _column_residual(observed, angle, manifold),
                start,
                bounds=(lower, upper),
                method="trf",
                max_nfev=DEFAULT_CONFIG.fit_max_evaluations,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            cost = float(np.dot(result.fun, result.fun))
            if np.isfinite(cost) and cost < best_cost:
                best_angle, best_cost = result.x, cost
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
    if best_angle is None:
        best_angle = np.asarray(starts[0], np.float64)
    return np.asarray(best_angle, np.float32), best_cost


def angles_from_mixing(
    mixing: np.ndarray, manifold: ManifoldFunction
) -> np.ndarray:
    def fit_column(observed: np.ndarray) -> np.ndarray:
        observed = np.asarray(observed, np.complex128)
        norm = np.linalg.norm(observed)
        if not np.isfinite(norm) or norm < 1e-10:
            return np.asarray([0.0, 42.5], np.float32)
        observed /= norm
        initial = nominal_initial_angle(observed)
        lower = [DEFAULT_CONFIG.azimuth_bounds_deg[0], DEFAULT_CONFIG.elevation_bounds_deg[0]]
        upper = [DEFAULT_CONFIG.azimuth_bounds_deg[1], DEFAULT_CONFIG.elevation_bounds_deg[1]]
        try:
            result = least_squares(
                lambda angle: _column_residual(observed, angle, manifold),
                initial,
                bounds=(lower, upper),
                method="trf",
                max_nfev=DEFAULT_CONFIG.fit_max_evaluations,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            return result.x.astype(np.float32)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return initial.astype(np.float32)

    return np.stack([fit_column(mixing[:, source]) for source in range(mixing.shape[1])])


def project_with_starts(
    mixing: np.ndarray,
    manifold: ManifoldFunction,
    starts: Sequence[Sequence[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles, objectives = [], []
    for source in range(mixing.shape[1]):
        angle, objective = fit_from_starts(mixing[:, source], manifold, starts[source])
        angles.append(angle)
        objectives.append(objective)
    angles_array = np.stack(angles, axis=0)
    columns = manifold(angles_array)
    projected = np.empty_like(columns)
    for source in range(mixing.shape[1]):
        denominator = max(float(np.vdot(columns[:, source], columns[:, source]).real), 1e-12)
        coefficient = np.vdot(columns[:, source], mixing[:, source]) / denominator
        projected[:, source] = coefficient * columns[:, source]
    return projected, angles_array, np.asarray(objectives, np.float64)


def coarse_angle_grid() -> np.ndarray:
    step = DEFAULT_CONFIG.coarse_grid_step_deg
    azimuth = np.arange(*DEFAULT_CONFIG.azimuth_bounds_deg, step)
    azimuth = np.append(azimuth, DEFAULT_CONFIG.azimuth_bounds_deg[1])
    elevation = np.arange(*DEFAULT_CONFIG.elevation_bounds_deg, step)
    elevation = np.append(elevation, DEFAULT_CONFIG.elevation_bounds_deg[1])
    azimuth_grid, elevation_grid = np.meshgrid(azimuth, elevation, indexing="xy")
    return np.stack([azimuth_grid.reshape(-1), elevation_grid.reshape(-1)], axis=1)


COARSE_GRID = coarse_angle_grid()
