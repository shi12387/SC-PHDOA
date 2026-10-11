"""Source-wise analytic and aligned-IQ feature extraction."""

from __future__ import annotations

import numpy as np


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, np.complex128)
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix / np.maximum(norms, 1e-10)


def aligned_iq_block(
    snapshots: np.ndarray,
    preamble: np.ndarray,
    delay: float,
    cfo: float,
    samples: int = 256,
) -> np.ndarray:
    length = min(samples, len(preamble))
    delay_index = int(np.clip(round(float(delay)), 0, snapshots.shape[1] - length))
    time = np.arange(length, dtype=np.float64)
    reference = preamble[:length] * np.exp(2j * np.pi * float(cfo) * time)
    block = snapshots[:, delay_index : delay_index + length] * np.conj(reference)[None]
    block *= np.exp(-1j * np.angle(np.mean(block[0])))
    scale = max(float(np.sqrt(np.mean(np.abs(block) ** 2))), 1e-8)
    return np.stack([block.real / scale, block.imag / scale], axis=0).astype(np.float32)


def analytic_source_features(
    mixing: np.ndarray,
    synchronization: np.ndarray,
    source_count: int,
    snr_db: float,
    maximum_delay: int,
    maximum_sources: int = 3,
) -> np.ndarray:
    """Return the exact 26-D feature vector used for session calibration."""
    output = np.zeros((maximum_sources, 26), np.float32)
    normalized = normalize_columns(mixing)
    norms = np.linalg.norm(mixing, axis=0)
    for source in range(source_count):
        column = normalized[:, source].copy()
        column *= np.exp(-1j * np.angle(column[0]))
        output[source, :21] = np.stack(
            [column.real, column.imag, np.abs(column)], axis=-1
        ).reshape(-1)
        output[source, 21] = source_count / maximum_sources
        output[source, 22] = np.clip(snr_db / 20.0, -0.25, 1.0)
        output[source, 23] = np.clip(
            synchronization[source, 0] / max(maximum_delay, 1), 0.0, 1.0
        )
        output[source, 24] = np.clip(synchronization[source, 1] / 0.01, -1.5, 1.5)
        output[source, 25] = np.clip(np.log(max(norms[source], 1e-6)) / 5.0, -1.0, 1.0)
    return output
