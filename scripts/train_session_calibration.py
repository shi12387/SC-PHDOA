"""Train the arm-aware cross-scene session-calibration network."""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.calibration import CalibrationInputAblationNetwork, SessionCalibrationNetwork
from sc_phdoa.calibration_training import (
    calibration_gate_targets,
    calibration_loss_terms,
    calibration_target,
    fixed_context_indices,
    stratified_state_split,
)
from sc_phdoa.dataset import coupling_pairs, read_iq, read_shared_preamble, sample_from_hdf5
from sc_phdoa.features import aligned_iq_block, analytic_source_features, normalize_columns
from sc_phdoa.synchronization import estimate_synchronization_and_mixing
from sc_phdoa.protocol import dataset_path, load_protocol, output_path

PROTOCOL = load_protocol()
TRAINING_MAT = dataset_path("session_calibration_training")
CACHE_DIRECTORY = output_path("session_calibration_cache")
OUTPUT_DIRECTORY = output_path("session_calibration_training")
SEED = int(PROTOCOL["random_seed"])
EPOCHS = 80
BATCH_SIZE = 8
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 2e-4
EARLY_STOPPING_PATIENCE = 16
CONTEXT_SCENES = int(PROTOCOL["context_scenes"])


def scalar(handle: h5py.File, name: str, default: float | None = None) -> float:
    if name not in handle:
        if default is None:
            raise KeyError(name)
        return float(default)
    return float(np.asarray(handle[name]).reshape(-1)[0])


def vector(handle: h5py.File, name: str, dtype: type) -> np.ndarray:
    return np.asarray(handle[name]).reshape(-1).astype(dtype)


def nominal_coordinates(handle: h5py.File) -> np.ndarray:
    value = np.asarray(handle["nominal_array_coords"], np.float64)
    if value.shape == (7, 3):
        value = value.T
    return value.reshape(3, 7)


def read_calibration(
    handle: h5py.File, index: int, sample_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gain_real = sample_from_hdf5(handle["sensor_gain_real"], index, sample_count)
    gain_imag = sample_from_hdf5(handle["sensor_gain_imag"], index, sample_count)
    gain = np.asarray(gain_real + 1j * gain_imag).reshape(-1)
    coupling_real = sample_from_hdf5(handle["coupling_real"], index, sample_count)
    coupling_imag = sample_from_hdf5(handle["coupling_imag"], index, sample_count)
    coupling = np.asarray(coupling_real + 1j * coupling_imag).reshape(7, 7)
    coordinates = np.asarray(
        sample_from_hdf5(handle["actual_array_coords_all"], index, sample_count),
        np.float64,
    )
    if coordinates.shape == (7, 3):
        coordinates = coordinates.T
    return gain, coupling, coordinates.reshape(3, 7)


def cache_paths(directory: Path) -> tuple[Path, Path]:
    return directory / "metadata.npz", directory / "aligned_iq_float16.npy"


def build_cache(mat_path: Path, directory: Path) -> tuple[Path, Path]:
    metadata_path, iq_path = cache_paths(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(mat_path, "r") as handle:
        counts = vector(handle, "num_sources_all", np.int64)
        snrs = vector(handle, "snr_all", np.float64)
        state_ids = vector(handle, "array_state_id_all", np.int64)
        scene_ids = vector(handle, "scene_within_state_all", np.int64)
        cases = vector(handle, "impairment_case_all", np.int64)
        states = np.unique(state_ids)
        sample_count = len(counts)
        scene_count = int(round(scalar(handle, "scenes_per_state", 18)))
        preamble = read_shared_preamble(handle)[:256]
        maximum_delay = int(round(scalar(handle, "max_timing_offset", 128)))
        wavelength = scalar(handle, "lambda_val")
        spacing = scalar(handle, "d")
        nominal = nominal_coordinates(handle)
        pairs = coupling_pairs(nominal, spacing)
        shape = (len(states), scene_count)
        features = np.zeros(shape + (3, 26), np.float32)
        source_mask = np.zeros(shape + (3,), bool)
        mixing_real = np.zeros(shape + (3, 7), np.float32)
        mixing_imag = np.zeros_like(mixing_real)
        originals = np.full(shape, -1, np.int64)
        state_counts = np.zeros(shape, np.int64)
        state_snrs = np.zeros(shape, np.float32)
        state_cases = np.zeros(len(states), np.int64)
        gain_real = np.zeros((len(states), 7), np.float32)
        gain_imag = np.zeros_like(gain_real)
        coupling_real = np.zeros((len(states), 7, 7), np.float32)
        coupling_imag = np.zeros_like(coupling_real)
        coordinates_all = np.zeros((len(states), 3, 7), np.float32)
        targets, gate_labels = [], []
        aligned = np.lib.format.open_memmap(
            iq_path,
            mode="w+",
            dtype=np.float16,
            shape=shape + (3, 2, 7, 256),
        )
        completed = 0
        for row, state_id in enumerate(states):
            indices = np.flatnonzero(state_ids == state_id)
            indices = indices[np.argsort(scene_ids[indices])]
            if len(indices) != scene_count:
                raise ValueError(f"state {state_id} has {len(indices)} scenes")
            state_cases[row] = cases[indices[0]]
            gain, coupling, coordinates = read_calibration(handle, int(indices[0]), sample_count)
            gain_real[row], gain_imag[row] = gain.real, gain.imag
            coupling_real[row], coupling_imag[row] = coupling.real, coupling.imag
            coordinates_all[row] = coordinates
            target = calibration_target(gain, coupling, coordinates, nominal, spacing, pairs)
            targets.append(target)
            gate_labels.append(calibration_gate_targets(target))
            for scene, original in enumerate(indices):
                count = int(counts[original])
                snapshots = read_iq(handle, int(original), sample_count)
                mixing, synchronization = estimate_synchronization_and_mixing(
                    snapshots, preamble, maximum_delay, count
                )
                features[row, scene] = analytic_source_features(
                    mixing, synchronization, count, snrs[original], maximum_delay
                )
                source_mask[row, scene, :count] = True
                normalized = normalize_columns(mixing)
                mixing_real[row, scene, :count] = normalized.T.real
                mixing_imag[row, scene, :count] = normalized.T.imag
                for source in range(count):
                    aligned[row, scene, source] = aligned_iq_block(
                        snapshots,
                        preamble,
                        synchronization[source, 0],
                        synchronization[source, 1],
                    ).astype(np.float16)
                originals[row, scene] = original
                state_counts[row, scene] = count
                state_snrs[row, scene] = snrs[original]
                completed += 1
                if completed % 100 == 0 or completed == sample_count:
                    print(f"synchronized scenes {completed}/{sample_count}")
        aligned.flush()
        del aligned
    np.savez_compressed(
        metadata_path,
        features=features,
        source_mask=source_mask,
        mixing_real=mixing_real,
        mixing_imag=mixing_imag,
        originals=originals,
        counts=state_counts,
        snrs=state_snrs,
        state_ids=states,
        state_cases=state_cases,
        gains_real=gain_real,
        gains_imag=gain_imag,
        couplings_real=coupling_real,
        couplings_imag=coupling_imag,
        coordinates=coordinates_all,
        targets=np.stack(targets),
        gate_targets=np.stack(gate_labels),
        nominal=nominal.astype(np.float32),
        spacing=np.asarray(spacing),
        wavelength=np.asarray(wavelength),
        pairs=np.asarray(pairs, np.int64),
    )
    return metadata_path, iq_path


def load_cache(mat_path: Path, directory: Path) -> dict[str, np.ndarray]:
    metadata_path, iq_path = cache_paths(directory)
    if not metadata_path.exists() or not iq_path.exists():
        build_cache(mat_path, directory)
    arrays = dict(np.load(metadata_path, allow_pickle=False))
    arrays["aligned_iq"] = np.load(iq_path, mmap_mode="r")
    return arrays


class CalibrationStateDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray) -> None:
        self.arrays = arrays
        self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        arrays = self.arrays
        return {
            "index": torch.tensor(index),
            "features": torch.from_numpy(arrays["features"][index]),
            "source_mask": torch.from_numpy(arrays["source_mask"][index]),
            "aligned_iq": torch.from_numpy(
                np.array(arrays["aligned_iq"][index], dtype=np.float32, copy=True)
            ),
            "target": torch.from_numpy(arrays["targets"][index]),
            "gate_target": torch.from_numpy(arrays["gate_targets"][index]),
            "gain_real": torch.from_numpy(arrays["gains_real"][index]),
            "gain_imag": torch.from_numpy(arrays["gains_imag"][index]),
            "coupling_real": torch.from_numpy(arrays["couplings_real"][index]),
            "coupling_imag": torch.from_numpy(arrays["couplings_imag"][index]),
            "coordinates": torch.from_numpy(arrays["coordinates"][index]),
        }


def gather_context(
    batch: dict[str, torch.Tensor], indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.arange(len(indices), device=indices.device)[:, None]
    return (
        batch["features"][rows, indices],
        batch["source_mask"][rows, indices],
        batch["aligned_iq"][rows, indices],
    )


def evaluate_validation(
    model: SessionCalibrationNetwork,
    loader: DataLoader,
    context_lookup: dict[int, torch.Tensor],
    device: torch.device,
    nominal: torch.Tensor,
    spacing: float,
    pairs: list[tuple[int, int]],
    wavelength: float,
    probes: torch.Tensor,
) -> np.ndarray:
    model.eval()
    accumulated = np.zeros(5)
    sample_count = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in raw_batch.items()}
            indices = torch.stack([
                context_lookup[int(index)] for index in batch["index"].cpu().numpy()
            ]).to(device)
            output = model(*gather_context(batch, indices))
            terms = calibration_loss_terms(
                output, batch, nominal, spacing, pairs, wavelength, probes
            )
            size = len(batch["index"])
            accumulated += np.asarray([term.item() for term in terms]) * size
            sample_count += size
    return accumulated / max(sample_count, 1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train session calibration")
    parser.add_argument("--mat", type=Path, default=TRAINING_MAT)
    parser.add_argument("--cache", type=Path, default=CACHE_DIRECTORY)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--variant",
        choices=("full", "analytic_only", "iq_only"),
        default="full",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    device = torch.device(arguments.device)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    arguments.output.mkdir(parents=True, exist_ok=True)
    arrays = load_cache(arguments.mat, arguments.cache)
    splits = stratified_state_split(arrays["state_cases"], SEED)
    train_loader = DataLoader(
        CalibrationStateDataset(arrays, splits["train"]),
        BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        CalibrationStateDataset(arrays, splits["validation"]),
        BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    fixed = fixed_context_indices(
        splits["validation"], arrays["features"].shape[1], CONTEXT_SCENES, SEED
    )
    context_lookup = {
        int(index): torch.from_numpy(fixed[row])
        for row, index in enumerate(splits["validation"])
    }
    pairs = [tuple(map(int, pair)) for pair in arrays["pairs"]]
    nominal = torch.from_numpy(arrays["nominal"]).float().to(device)
    spacing = float(arrays["spacing"])
    wavelength = float(arrays["wavelength"])
    probes = torch.tensor(
        [[azimuth, elevation] for elevation in (10.0, 30.0, 50.0, 70.0)
         for azimuth in (-50.0, -25.0, 0.0, 25.0, 50.0)],
        dtype=torch.float32,
        device=device,
    )
    model = (
        SessionCalibrationNetwork()
        if arguments.variant == "full"
        else CalibrationInputAblationNetwork(arguments.variant)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    checkpoint_path = arguments.output / "network.pt"
    history = []
    best_loss = float("inf")
    stale_epochs = 0
    scene_count = arrays["features"].shape[1]
    for epoch in range(1, EPOCHS + 1):
        model.train()
        accumulated = np.zeros(5)
        samples = 0
        started = time.perf_counter()
        for raw_batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in raw_batch.items()}
            context = torch.topk(
                torch.rand(len(batch["index"]), scene_count, device=device),
                CONTEXT_SCENES,
                dim=1,
            ).indices
            output = model(*gather_context(batch, context))
            terms = calibration_loss_terms(
                output, batch, nominal, spacing, pairs, wavelength, probes
            )
            optimizer.zero_grad(set_to_none=True)
            terms[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            size = len(batch["index"])
            accumulated += np.asarray([term.item() for term in terms]) * size
            samples += size
        training = accumulated / max(samples, 1)
        validation = evaluate_validation(
            model, validation_loader, context_lookup, device,
            nominal, spacing, pairs, wavelength, probes,
        )
        scheduler.step(validation[0])
        row = {
            "epoch": epoch,
            "train_loss": training[0],
            "validation_loss": validation[0],
            "train_parameter_loss": training[1],
            "validation_parameter_loss": validation[1],
            "train_manifold_loss": training[2],
            "validation_manifold_loss": validation[2],
            "train_gate_loss": training[3],
            "validation_gate_loss": validation[3],
            "train_identity_loss": training[4],
            "validation_identity_loss": validation[4],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}: train={training[0]:.6f}, "
            f"validation={validation[0]:.6f}"
        )
        if validation[0] < best_loss - 1e-5:
            best_loss = float(validation[0])
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_score": best_loss,
                    "output_dim": 50,
                    "context_scenes": CONTEXT_SCENES,
                    "model_parameters": sum(p.numel() for p in model.parameters()),
                    "variant": arguments.variant,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                print("early stopping")
                break
    with (arguments.output / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(arguments.output / "data_split.npz", **splits)
    print(f"Saved: {arguments.output}")


if __name__ == "__main__":
    main()
