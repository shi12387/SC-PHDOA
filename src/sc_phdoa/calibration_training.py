"""Losses, targets, and deterministic splits for session-calibration training."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .config import DEFAULT_CONFIG


def calibration_target(
    gain: np.ndarray,
    coupling: np.ndarray,
    coordinates: np.ndarray,
    nominal_coordinates: np.ndarray,
    spacing: float,
    coupling_pairs: list[tuple[int, int]],
) -> np.ndarray:
    gain_delta = gain[1:] - 1.0
    position = (
        (coordinates - nominal_coordinates)
        / spacing
        / DEFAULT_CONFIG.position_scale_spacing
    )
    values = list(gain_delta.real / DEFAULT_CONFIG.gain_scale)
    values += list(gain_delta.imag / DEFAULT_CONFIG.gain_scale)
    values += list(position[0, 1:]) + list(position[2, 1:])
    values += [
        coupling[first, second].real / DEFAULT_CONFIG.coupling_scale
        for first, second in coupling_pairs
    ]
    values += [
        coupling[first, second].imag / DEFAULT_CONFIG.coupling_scale
        for first, second in coupling_pairs
    ]
    return np.clip(np.asarray(values, np.float32), -1.0, 1.0)


def calibration_gate_targets(target: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            float(np.linalg.norm(target[:12]) > 1e-5),
            float(np.linalg.norm(target[12:24]) > 1e-5),
            float(np.linalg.norm(target[24:]) > 1e-5),
        ],
        np.float32,
    )


def stratified_state_split(cases: np.ndarray, seed: int = 2026) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    output: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for case in sorted(np.unique(cases)):
        indices = np.flatnonzero(cases == case)
        rng.shuffle(indices)
        train_count = int(round(0.60 * len(indices)))
        validation_count = int(round(0.20 * len(indices)))
        output["train"].extend(indices[:train_count])
        output["validation"].extend(indices[train_count : train_count + validation_count])
        output["test"].extend(indices[train_count + validation_count :])
    return {
        name: np.asarray(sorted(indices), np.int64)
        for name, indices in output.items()
    }


def fixed_context_indices(
    state_indices: np.ndarray,
    scene_count: int,
    context_count: int = 8,
    seed: int = 2026,
) -> np.ndarray:
    output = np.zeros((len(state_indices), context_count), np.int64)
    for row, state in enumerate(state_indices):
        rng = np.random.default_rng(seed + 7919 * int(state))
        output[row] = np.sort(rng.choice(scene_count, context_count, replace=False))
    return output


def decode_calibration_torch(
    values: torch.Tensor,
    nominal_coordinates: torch.Tensor,
    spacing: float,
    coupling_pairs: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = values.shape[0]
    cursor = 0
    real = values[:, cursor : cursor + 6] * DEFAULT_CONFIG.gain_scale
    cursor += 6
    imag = values[:, cursor : cursor + 6] * DEFAULT_CONFIG.gain_scale
    cursor += 6
    gain = torch.ones(batch, 7, dtype=torch.complex64, device=values.device)
    gain[:, 1:] += torch.complex(real, imag)
    coordinates = nominal_coordinates.unsqueeze(0).repeat(batch, 1, 1)
    scale = DEFAULT_CONFIG.position_scale_spacing * spacing
    coordinates[:, 0, 1:] += values[:, cursor : cursor + 6] * scale
    cursor += 6
    coordinates[:, 2, 1:] += values[:, cursor : cursor + 6] * scale
    cursor += 6
    pair_count = len(coupling_pairs)
    real = values[:, cursor : cursor + pair_count] * DEFAULT_CONFIG.coupling_scale
    cursor += pair_count
    imag = values[:, cursor : cursor + pair_count] * DEFAULT_CONFIG.coupling_scale
    coupling = torch.eye(
        7, dtype=torch.complex64, device=values.device
    ).unsqueeze(0).repeat(batch, 1, 1)
    for index, (first, second) in enumerate(coupling_pairs):
        value = torch.complex(real[:, index], imag[:, index])
        coupling[:, first, second] = value
        coupling[:, second, first] = value
    return gain, coupling, coordinates


def manifold_tensor(
    gain: torch.Tensor,
    coupling: torch.Tensor,
    coordinates: torch.Tensor,
    wavelength: float,
    angles: torch.Tensor,
) -> torch.Tensor:
    azimuth = torch.deg2rad(angles[:, 0])
    elevation = torch.deg2rad(angles[:, 1])
    ux = torch.cos(elevation) * torch.sin(azimuth)
    uz = torch.sin(elevation)
    phase = (
        coordinates[:, 0, :, None] * ux[None, None]
        + coordinates[:, 2, :, None] * uz[None, None]
    )
    steering = torch.exp(
        torch.complex(torch.zeros_like(phase), -2.0 * math.pi * phase / wavelength)
    )
    return torch.matmul(coupling, gain[:, :, None] * steering)


def calibration_loss_terms(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    nominal_coordinates: torch.Tensor,
    spacing: float,
    coupling_pairs: list[tuple[int, int]],
    wavelength: float,
    probe_angles: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    parameter = nn.functional.smooth_l1_loss(output["final"], batch["target"])
    predicted = manifold_tensor(
        *decode_calibration_torch(
            output["final"], nominal_coordinates, spacing, coupling_pairs
        ),
        wavelength,
        probe_angles,
    )
    gain = torch.complex(batch["gain_real"], batch["gain_imag"])
    coupling = torch.complex(batch["coupling_real"], batch["coupling_imag"])
    truth = manifold_tensor(gain, coupling, batch["coordinates"], wavelength, probe_angles)
    numerator = torch.abs(torch.sum(torch.conj(predicted) * truth, dim=1)) ** 2
    denominator = (
        torch.sum(torch.abs(predicted) ** 2, dim=1)
        * torch.sum(torch.abs(truth) ** 2, dim=1)
    ).clamp_min(1e-8)
    physical = (1.0 - numerator / denominator).mean()
    gate = nn.functional.binary_cross_entropy(output["gates"], batch["gate_target"])
    ideal = batch["gate_target"].sum(dim=1) == 0
    identity = (
        output["final"][ideal].square().mean()
        if torch.any(ideal)
        else output["final"].sum() * 0.0
    )
    total = parameter + 2.0 * physical + 0.15 * gate + 0.50 * identity
    return total, parameter, physical, gate, identity
