"""External-session MAT-file indexing and protocol validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


def _scalar(handle: h5py.File, name: str, default: float | None = None) -> float:
    if name not in handle:
        if default is None:
            raise KeyError(name)
        return float(default)
    return float(np.asarray(handle[name]).reshape(-1)[0])


def _vector(handle: h5py.File, name: str, dtype: type) -> np.ndarray:
    return np.asarray(handle[name]).reshape(-1).astype(dtype)


def _text(handle: h5py.File, name: str) -> str:
    if name not in handle:
        return ""
    value = np.asarray(handle[name])
    if value.dtype.kind in "ui":
        return "".join(chr(int(item)) for item in value.ravel(order="F") if int(item)).strip()
    if value.dtype.kind in "SU":
        return "".join(value.astype(str).ravel(order="F")).strip()
    return str(value.squeeze())


def sample_from_hdf5(dataset: h5py.Dataset, index: int, sample_count: int) -> np.ndarray:
    axes = [axis for axis, size in enumerate(dataset.shape) if size == sample_count]
    if not axes:
        raise ValueError(f"cannot locate sample axis in {dataset.name}: {dataset.shape}")
    return np.asarray(np.take(dataset, index, axis=axes[-1]))


def read_iq(handle: h5py.File, index: int, sample_count: int) -> np.ndarray:
    value = sample_from_hdf5(handle["IQ_tensors"], index, sample_count)
    if value.shape[0] == 2:
        value = value[0] + 1j * value[1]
    elif value.shape[-1] == 2:
        value = value[..., 0] + 1j * value[..., 1]
    if value.shape[0] != 7 and value.shape[1] == 7:
        value = value.T
    if value.shape[0] != 7:
        raise ValueError(f"unexpected IQ shape: {value.shape}")
    return np.asarray(value, np.complex128)


def read_shared_preamble(handle: h5py.File) -> np.ndarray:
    real = np.asarray(handle["shared_preamble_real"]).reshape(-1)
    imag = np.asarray(handle["shared_preamble_imag"]).reshape(-1)
    return np.asarray(real + 1j * imag, np.complex128)


def read_doas(
    handle: h5py.File, index: int, sample_count: int, source_count: int
) -> np.ndarray:
    value = sample_from_hdf5(handle["doas_all"], index, sample_count)
    if value.shape == (2, 3):
        value = value.T
    value = np.asarray(value, np.float64).reshape(3, 2)
    return value[:source_count]


def read_calibration(
    handle: h5py.File, index: int, sample_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gain = sample_from_hdf5(handle["sensor_gain_real"], index, sample_count)
    gain = np.asarray(gain) + 1j * sample_from_hdf5(
        handle["sensor_gain_imag"], index, sample_count
    )
    coupling = sample_from_hdf5(handle["coupling_real"], index, sample_count)
    coupling = np.asarray(coupling) + 1j * sample_from_hdf5(
        handle["coupling_imag"], index, sample_count
    )
    coordinates = np.asarray(
        sample_from_hdf5(handle["actual_array_coords_all"], index, sample_count),
        np.float64,
    )
    if coordinates.shape == (7, 3):
        coordinates = coordinates.T
    return (
        np.asarray(gain, np.complex128).reshape(7),
        np.asarray(coupling, np.complex128).reshape(7, 7),
        coordinates.reshape(3, 7),
    )


def coupling_pairs(coordinates: np.ndarray, spacing: float) -> list[tuple[int, int]]:
    return [
        (first, second)
        for first in range(7)
        for second in range(first + 1, 7)
        if np.linalg.norm(coordinates[:, first] - coordinates[:, second]) / spacing <= 2.25
    ]


@dataclass(frozen=True)
class SessionDataset:
    path: Path
    original_indices: np.ndarray
    state_ids: np.ndarray
    source_counts: np.ndarray
    snrs: np.ndarray
    roles: np.ndarray
    impairment_cases: np.ndarray
    context_scenes: np.ndarray
    guard_scenes: np.ndarray
    target_scenes: np.ndarray
    nominal_coordinates: np.ndarray
    spacing: float
    wavelength: float
    coupling_pairs: tuple[tuple[int, int], ...]
    preamble: np.ndarray
    maximum_delay: int
    sample_count: int

    @classmethod
    def load_external(
        cls, path: Path, validate_identity: bool = True
    ) -> "SessionDataset":
        path = Path(path)
        with h5py.File(path, "r") as handle:
            counts = _vector(handle, "num_sources_all", np.int64)
            snrs = _vector(handle, "snr_all", np.float64)
            state_ids_all = _vector(handle, "array_state_id_all", np.int64)
            scene_ids = _vector(handle, "scene_within_state_all", np.int64)
            cases = _vector(handle, "impairment_case_all", np.int64)
            roles_all = _vector(handle, "evaluation_role_all", np.int64)
            states = np.unique(state_ids_all)
            scene_count = int(round(_scalar(handle, "scenes_per_state", 18)))
            nominal = np.asarray(handle["nominal_array_coords"], np.float64)
            if nominal.shape == (7, 3):
                nominal = nominal.T
            nominal = nominal.reshape(3, 7)
            spacing = _scalar(handle, "d")
            wavelength = _scalar(handle, "lambda_val")
            shape = (len(states), scene_count)
            originals = np.full(shape, -1, np.int64)
            state_counts = np.zeros(shape, np.int64)
            state_snrs = np.zeros(shape, np.float32)
            state_roles = np.zeros(shape, np.int64)
            state_cases = np.zeros(len(states), np.int64)
            context = np.zeros((len(states), 8), np.int64)
            guard = np.zeros((len(states), 4), np.int64)
            target = np.zeros((len(states), 6), np.int64)
            for row, state_id in enumerate(states):
                indices = np.flatnonzero(state_ids_all == state_id)
                indices = indices[np.argsort(scene_ids[indices])]
                if len(indices) != scene_count or len(np.unique(cases[indices])) != 1:
                    raise ValueError(f"invalid external session {state_id}")
                originals[row], state_counts[row] = indices, counts[indices]
                state_snrs[row], state_roles[row] = snrs[indices], roles_all[indices]
                state_cases[row] = cases[indices[0]]
                context[row] = np.flatnonzero(state_roles[row] == 0)
                guard[row] = np.flatnonzero(state_roles[row] == 1)
                target[row] = np.flatnonzero(state_roles[row] == 2)
            metadata = {
                "dataset_role": _text(handle, "dataset_role"),
                "generator_seed": int(round(_scalar(handle, "generator_seed", -1))),
                "preamble_length": int(round(_scalar(handle, "preamble_len", -1))),
                "maximum_delay": int(round(_scalar(handle, "max_timing_offset", -1))),
                "snapshot_length": read_iq(handle, 0, len(counts)).shape[1],
            }
            expected = {
                "dataset_role": "never_observed_external_paper_test",
                "generator_seed": 18020,
                "preamble_length": 256,
                "maximum_delay": 128,
                "snapshot_length": 1024,
            }
            failed = (
                [key for key, value in expected.items() if metadata[key] != value]
                if validate_identity
                else []
            )
            if scene_count != 18:
                failed.append("scene count")
            if np.any(np.sum(state_roles == 0, axis=1) != 8):
                failed.append("context roles")
            if np.any(np.sum(state_roles == 1, axis=1) != 4):
                failed.append("guard roles")
            if np.any(np.sum(state_roles == 2, axis=1) != 6):
                failed.append("target roles")
            if validate_identity and (len(states) != 300 or len(counts) != 5400):
                failed.append("external session dimensions")
            if failed:
                raise RuntimeError("external protocol validation failed: " + ", ".join(failed))
            preamble = read_shared_preamble(handle)[:256]
        return cls(
            path=path,
            original_indices=originals,
            state_ids=states.astype(np.int64),
            source_counts=state_counts,
            snrs=state_snrs,
            roles=state_roles,
            impairment_cases=state_cases,
            context_scenes=context,
            guard_scenes=guard,
            target_scenes=target,
            nominal_coordinates=nominal.astype(np.float64),
            spacing=float(spacing),
            wavelength=float(wavelength),
            coupling_pairs=tuple(coupling_pairs(nominal, spacing)),
            preamble=preamble,
            maximum_delay=metadata["maximum_delay"],
            sample_count=len(counts),
        )

    def read_session(self, state_index: int) -> list[np.ndarray]:
        with h5py.File(self.path, "r") as handle:
            return [
                read_iq(handle, int(index), self.sample_count)
                for index in self.original_indices[state_index]
            ]
