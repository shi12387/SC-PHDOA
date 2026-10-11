"""Shared-preamble delay/CFO estimation and source mixing recovery."""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_CONFIG


def ambiguity_surface(
    snapshots: np.ndarray, preamble: np.ndarray, maximum_delay: int
) -> tuple[np.ndarray, np.ndarray]:
    length = len(preamble)
    if snapshots.shape[1] < maximum_delay + length:
        raise ValueError("snapshot length is shorter than delay plus preamble")
    windows = np.lib.stride_tricks.sliding_window_view(
        snapshots, length, axis=1
    )[:, : maximum_delay + 1, :]
    products = windows * np.conj(preamble)[None, None, :]
    spectrum = np.fft.fft(products, n=DEFAULT_CONFIG.sync_fft_size, axis=-1)
    frequencies = np.fft.fftfreq(DEFAULT_CONFIG.sync_fft_size)
    low, high = DEFAULT_CONFIG.cfo_search_bounds
    keep = (frequencies >= low) & (frequencies <= high)
    return spectrum[:, :, keep], frequencies[keep]


def synchronization_template(
    preamble: np.ndarray, delay: float, cfo: float, total_length: int
) -> np.ndarray:
    delay_index = int(round(float(delay)))
    time = np.arange(len(preamble), dtype=np.float64)
    atom = np.zeros(total_length, np.complex128)
    atom[delay_index : delay_index + len(preamble)] = preamble * np.exp(
        2j * np.pi * float(cfo) * time
    )
    return atom


def estimate_mixing(
    snapshots: np.ndarray, templates: np.ndarray, ridge_relative: float = 1e-5
) -> np.ndarray:
    templates = np.asarray(templates, np.complex128)
    snapshots = np.asarray(snapshots, np.complex128)
    gram = templates @ templates.conj().T
    scale = max(float(np.trace(gram).real) / len(templates), 1e-12)
    inverse = np.linalg.inv(gram + ridge_relative * scale * np.eye(len(templates)))
    return snapshots @ templates.conj().T @ inverse


def _best_unexcluded_peak(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    selected: list[tuple[int, float]],
) -> tuple[int, float]:
    score = np.sum(np.abs(spectrum) ** 2, axis=0)
    delay_axis = np.arange(score.shape[0])[:, None]
    for delay, cfo in selected:
        suppress = (
            (np.abs(delay_axis - delay) <= DEFAULT_CONFIG.nms_delay)
            & (np.abs(frequencies[None, :] - cfo) <= DEFAULT_CONFIG.nms_cfo)
        )
        score[suppress] = -np.inf
    flat = int(np.argmax(score))
    delay_index, frequency_index = np.unravel_index(flat, score.shape)
    frequency = float(frequencies[frequency_index])
    if 0 < frequency_index < score.shape[1] - 1:
        values = np.log(
            np.maximum(score[delay_index, frequency_index - 1 : frequency_index + 2], 1e-20)
        )
        left, center, right = values
        denominator = left - 2.0 * center + right
        if np.isfinite(denominator) and abs(denominator) > 1e-12:
            offset = np.clip(0.5 * (left - right) / denominator, -0.5, 0.5)
            frequency += float(offset) * float(frequencies[1] - frequencies[0])
    return int(delay_index), frequency


def estimate_synchronization_and_mixing(
    snapshots: np.ndarray,
    preamble: np.ndarray,
    maximum_delay: int,
    source_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected: list[tuple[int, float]] = []
    residual = np.asarray(snapshots, np.complex128).copy()
    mixing = None
    for _ in range(source_count):
        spectrum, frequencies = ambiguity_surface(residual, preamble, maximum_delay)
        selected.append(_best_unexcluded_peak(spectrum, frequencies, selected))
        templates = np.stack(
            [
                synchronization_template(preamble, delay, cfo, snapshots.shape[1])
                for delay, cfo in selected
            ],
            axis=0,
        )
        mixing = estimate_mixing(snapshots, templates)
        residual = snapshots - mixing @ templates
    if mixing is None:
        raise ValueError("source_count must be positive")
    return mixing, np.asarray(selected, np.float64)
