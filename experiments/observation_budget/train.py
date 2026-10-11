"""Train validation-selected Cov/IQ-SetNet checkpoints for 8 or 12 supports."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.baselines.neural_set import NeuralSetBaseline
from sc_phdoa.calibration_training import stratified_state_split
from sc_phdoa.dataset import read_doas, read_iq
from sc_phdoa.metrics import ospa
from sc_phdoa.protocol import dataset_path, output_path

MAT_PATH = dataset_path("session_calibration_training")
SEED = 2026
EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4


def vector(handle: h5py.File, name: str, dtype) -> np.ndarray:
    return np.asarray(handle[name]).reshape(-1).astype(dtype)


def cache_paths(directory: Path) -> tuple[Path, Path]:
    return directory / "metadata.npz", directory / "raw_iq_float16.npy"


def build_cache(directory: Path) -> dict[str, np.ndarray]:
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path, iq_path = cache_paths(directory)
    if metadata_path.is_file() and iq_path.is_file():
        arrays = dict(np.load(metadata_path, allow_pickle=False))
        arrays["raw_iq"] = np.load(iq_path, mmap_mode="r")
        return arrays
    with h5py.File(MAT_PATH, "r") as handle:
        counts_all = vector(handle, "num_sources_all", np.int64)
        snrs_all = vector(handle, "snr_all", np.float64)
        state_ids_all = vector(handle, "array_state_id_all", np.int64)
        scene_ids_all = vector(handle, "scene_within_state_all", np.int64)
        cases_all = vector(handle, "impairment_case_all", np.int64)
        states = np.unique(state_ids_all)
        sample_count = len(counts_all)
        scene_count = int(np.asarray(handle["scenes_per_state"]).reshape(-1)[0])
        sensors, length = read_iq(handle, 0, sample_count).shape
        shape = (len(states), scene_count)
        originals = np.full(shape, -1, np.int64)
        counts = np.zeros(shape, np.int64)
        snrs = np.zeros(shape, np.float32)
        cases = np.zeros(len(states), np.int64)
        angles = np.full(shape + (3, 2), np.nan, np.float32)
        covariances = np.zeros(shape + (2, 7, 7), np.float32)
        log_power = np.zeros(shape, np.float32)
        raw = np.lib.format.open_memmap(iq_path, mode="w+", dtype=np.float16, shape=shape + (2, sensors, length))
        for state, state_id in enumerate(states):
            indices = np.flatnonzero(state_ids_all == state_id)
            indices = indices[np.argsort(scene_ids_all[indices])]
            cases[state] = int(cases_all[indices[0]])
            for scene, original_value in enumerate(indices):
                original = int(original_value)
                snapshots = np.asarray(read_iq(handle, original, sample_count), np.complex64)
                centered = snapshots - snapshots.mean(axis=1, keepdims=True)
                covariance = centered @ centered.conj().T / length
                power = max(float(np.trace(covariance).real / 7.0), 1e-10)
                covariance /= power
                raw[state, scene, 0], raw[state, scene, 1] = snapshots.real, snapshots.imag
                covariances[state, scene] = np.stack([covariance.real, covariance.imag])
                log_power[state, scene] = math.log(power)
                originals[state, scene] = original
                counts[state, scene] = int(counts_all[original])
                snrs[state, scene] = float(snrs_all[original])
                angles[state, scene, :counts[state, scene]] = read_doas(handle, original, sample_count, counts[state, scene])
            if (state + 1) % 25 == 0:
                print(f"SetNet cache: {state + 1}/{len(states)} states")
        raw.flush()
        del raw
    np.savez_compressed(metadata_path, originals=originals, counts=counts, snrs=snrs, state_ids=states, state_cases=cases, angles=angles, covariances=covariances, log_power=log_power)
    arrays = dict(np.load(metadata_path, allow_pickle=False))
    arrays["raw_iq"] = np.load(iq_path, mmap_mode="r")
    return arrays


class TrainingExamples(Dataset):
    def __init__(self, arrays, states, representation, support_count, power_mean, power_std, random_support):
        self.arrays, self.representation = arrays, representation
        self.support_count, self.power_mean, self.power_std = support_count, power_mean, power_std
        self.random_support = random_support
        self.entries = [(int(state), scene) for state in states for scene in range(arrays["counts"].shape[1])]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, item):
        state, target = self.entries[item]
        available = np.asarray([scene for scene in range(18) if scene != target], np.int64)
        if self.random_support:
            support = np.random.choice(available, self.support_count, replace=False)
        else:
            support = np.random.default_rng(SEED + 104729 * state + 1009 * target).choice(available, self.support_count, replace=False)
        scenes = np.concatenate([np.sort(support), [target]])
        if self.representation == "cov":
            signal = np.asarray(self.arrays["covariances"][state, scenes], np.float32)
        else:
            signal = np.asarray(self.arrays["raw_iq"][state, scenes], np.float32)
            signal /= np.sqrt(np.mean(signal * signal, axis=(1, 2, 3), keepdims=True) + 1e-8)
        power = (self.arrays["log_power"][state, scenes] - self.power_mean) / self.power_std
        return {
            "signal": torch.from_numpy(np.array(signal, copy=True)),
            "power": torch.from_numpy(power[:, None].astype(np.float32)),
            "angles": torch.from_numpy(np.array(self.arrays["angles"][state, target], copy=True)),
            "count": torch.tensor(int(self.arrays["counts"][state, target])),
        }


def direction(angles: torch.Tensor) -> torch.Tensor:
    azimuth, elevation = torch.deg2rad(angles[..., 0]), torch.deg2rad(angles[..., 1])
    return torch.stack([torch.cos(elevation) * torch.sin(azimuth), torch.cos(elevation) * torch.cos(azimuth), torch.sin(elevation)], dim=-1)


def loss_terms(output, batch):
    prediction = direction(output["angles"])
    exist_target = torch.zeros_like(output["exist_logits"])
    direction_losses = []
    for row in range(len(prediction)):
        count = int(batch["count"][row])
        cost = 1.0 - prediction[row] @ direction(batch["angles"][row, :count]).T
        queries, targets = linear_sum_assignment(cost.detach().cpu().numpy())
        queries = torch.as_tensor(queries, device=prediction.device)
        targets = torch.as_tensor(targets, device=prediction.device)
        direction_losses.append(cost[queries, targets].mean())
        exist_target[row, queries] = 1.0
    direction_loss = torch.stack(direction_losses).mean()
    existence = nn.functional.binary_cross_entropy_with_logits(output["exist_logits"], exist_target)
    count_loss = nn.functional.cross_entropy(output["count_logits"], batch["count"] - 1)
    cardinality = nn.functional.smooth_l1_loss(torch.sigmoid(output["exist_logits"]).sum(1), batch["count"].float())
    return direction_loss + 0.30 * existence + 0.20 * count_loss + 0.05 * cardinality


@torch.no_grad()
def validation_ospa(model, loader, device, use_amp):
    model.eval()
    values = []
    for batch in loader:
        moved = {key: value.to(device) for key, value in batch.items()}
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(moved["signal"], moved["power"])
        for row in range(len(moved["count"])):
            count = int(torch.argmax(output["count_logits"][row]).item()) + 1
            queries = torch.topk(torch.sigmoid(output["exist_logits"][row]), count).indices
            estimate = output["angles"][row, queries].cpu().numpy()
            truth_count = int(moved["count"][row])
            truth = moved["angles"][row, :truth_count].cpu().numpy()
            values.append(ospa(estimate, truth))
    return float(np.mean(values))


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-count", type=int, choices=(8, 12), required=True)
    parser.add_argument("--representations", nargs="+", choices=("cov", "iq"), default=("cov", "iq"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = arguments()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    cache = output_path("controlled_setnet_cache")
    arrays = build_cache(cache)
    splits = stratified_state_split(arrays["state_cases"], SEED)
    power_values = arrays["log_power"][splits["train"]].reshape(-1)
    power_mean, power_std = float(np.mean(power_values)), max(float(np.std(power_values)), 1e-4)
    destination = ROOT / "checkpoints" / "controlled_setnets" / f"support{args.support_count}"
    destination.mkdir(parents=True, exist_ok=True)
    for representation in args.representations:
        train_data = TrainingExamples(arrays, splits["train"], representation, args.support_count, power_mean, power_std, True)
        validation_data = TrainingExamples(arrays, splits["validation"], representation, args.support_count, power_mean, power_std, False)
        batch_size = 64 if representation == "cov" else 8
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0)
        torch.manual_seed(SEED + (0 if representation == "cov" else 10000))
        model = NeuralSetBaseline(representation).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=2e-6)
        use_amp = str(args.device).startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        best, stale, history = float("inf"), 0, []
        checkpoint_path = destination / ("covariance.pt" if representation == "cov" else "iq.pt")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running, samples = 0.0, 0
            for batch in train_loader:
                moved = {key: value.to(args.device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = loss_terms(model(moved["signal"], moved["power"]), moved)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
                running += float(loss.item()) * len(moved["count"]); samples += len(moved["count"])
            score = validation_ospa(model, validation_loader, args.device, use_amp)
            scheduler.step()
            history.append({"epoch": epoch, "training_loss": running / samples, "validation_ospa": score, "learning_rate": optimizer.param_groups[0]["lr"]})
            print(f"{representation} support={args.support_count} epoch={epoch}: validation OSPA={score:.4f}")
            if score < best - 1e-4:
                best, stale = score, 0
                torch.save({
                    "model": model.state_dict(), "representation": representation,
                    "epoch": epoch, "score": score, "phase": "selection",
                    "power_mean": power_mean, "power_std": power_std,
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "context_scenes": args.support_count, "token_dim": 128,
                    "encoder_layers": 2, "decoder_layers": 2, "attention_heads": 4,
                    "loss_weights": {"existence": 0.30, "count": 0.20, "cardinality": 0.05},
                }, checkpoint_path)
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        with (destination / f"{representation}_training_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0]); writer.writeheader(); writer.writerows(history)


if __name__ == "__main__":
    main()
