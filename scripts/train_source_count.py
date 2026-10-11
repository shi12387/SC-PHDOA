"""Train and select the 79-D source-count classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.dataset import read_iq, read_shared_preamble
from sc_phdoa.source_count import extract_source_count_features
from sc_phdoa.protocol import dataset_path, load_protocol, output_path

PROTOCOL = load_protocol()
TRAINING_MAT = dataset_path("source_count_training")
OUTPUT_DIRECTORY = output_path("source_count_training")
SEED = int(PROTOCOL["random_seed"])
VALIDATION_FRACTION = 0.20
TEST_FRACTION = 0.20


def vector(handle: h5py.File, name: str, dtype: type) -> np.ndarray:
    return np.asarray(handle[name]).reshape(-1).astype(dtype)


def geometry_splits(labels: np.ndarray, geometries: np.ndarray) -> dict[str, set[int]]:
    rng = np.random.default_rng(SEED)
    split = {"train": set(), "validation": set(), "test": set()}
    for source_count in (1, 2, 3):
        unique = np.unique(geometries[labels == source_count]).astype(np.int64)
        rng.shuffle(unique)
        test_count = max(1, int(round(len(unique) * TEST_FRACTION)))
        validation_count = max(1, int(round(len(unique) * VALIDATION_FRACTION)))
        split["test"].update(unique[:test_count].tolist())
        split["validation"].update(unique[test_count : test_count + validation_count].tolist())
        split["train"].update(unique[test_count + validation_count :].tolist())
    return split


def indices_for(split: set[int], geometries: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(geometries, np.asarray(sorted(split))))


def feature_columns(names: list[str], family: str) -> np.ndarray:
    if family == "covariance":
        return np.asarray([i for i, name in enumerate(names) if name.startswith(("full_", "payload_", "cross_"))])
    if family == "covariance_temporal":
        return np.asarray([i for i, name in enumerate(names) if not name.startswith("sync_")])
    if family == "all":
        return np.arange(len(names))
    raise ValueError(family)


def candidates():
    for regularization in (0.1, 1.0, 10.0):
        yield f"logistic_{regularization:g}", Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=regularization, max_iter=1000, random_state=SEED)),
        ])
    for leaves in (7, 15, 31):
        for minimum_samples in (20, 40):
            for regularization in (0.1, 1.0):
                yield f"histogram_boosting_{leaves}_{minimum_samples}_{regularization:g}", HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=250,
                    max_leaf_nodes=leaves,
                    min_samples_leaf=minimum_samples,
                    l2_regularization=regularization,
                    early_stopping=False,
                    random_state=SEED,
                )


def selection_score(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, dict[str, float]]:
    confusion = np.zeros((3, 3), np.int64)
    for truth, prediction in zip(labels, predictions):
        confusion[int(truth) - 1, int(prediction) - 1] += 1
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    accuracy = float(np.mean(labels == predictions))
    mean_absolute_error = float(np.mean(np.abs(labels - predictions)))
    score = accuracy + 0.10 * float(np.mean(recalls)) + 0.10 * float(np.min(recalls)) - 0.05 * mean_absolute_error
    return score, {
        "accuracy": accuracy,
        "macro_recall": float(np.mean(recalls)),
        "minimum_recall": float(np.min(recalls)),
        "mean_absolute_error": mean_absolute_error,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the source-count estimator")
    parser.add_argument("--mat", type=Path, default=TRAINING_MAT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(arguments.mat, "r") as handle:
        labels = vector(handle, "num_sources_all", np.int64)
        geometries = vector(handle, "geometry_id_all", np.int64)
        preamble = read_shared_preamble(handle)
        maximum_delay = int(round(vector(handle, "max_timing_offset", np.float64)[0]))
        sample_count = len(labels)
        snapshot_length = read_iq(handle, 0, sample_count).shape[1]
        payload_start = min(maximum_delay + len(preamble), snapshot_length - 16)
        features = []
        feature_names = None
        for index in range(sample_count):
            current, names = extract_source_count_features(
                read_iq(handle, index, sample_count), payload_start, preamble, maximum_delay
            )
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise RuntimeError("source-count feature order changed between samples")
            features.append(current)
            if (index + 1) % 100 == 0 or index + 1 == sample_count:
                print(f"features {index + 1}/{sample_count}")
    features_array = np.stack(features)
    split = geometry_splits(labels, geometries)
    train_indices = indices_for(split["train"], geometries)
    validation_indices = indices_for(split["validation"], geometries)
    test_indices = indices_for(split["test"], geometries)
    best = None
    selection_rows = []
    for family in ("covariance", "covariance_temporal", "all"):
        columns = feature_columns(feature_names, family)
        for name, estimator in candidates():
            estimator.fit(features_array[train_indices][:, columns], labels[train_indices])
            predictions = estimator.predict(features_array[validation_indices][:, columns])
            score, metrics = selection_score(labels[validation_indices], predictions)
            row = {"family": family, "model": name, "features": len(columns), "score": score, **metrics}
            selection_rows.append(row)
            if best is None or score > best["score"]:
                best = {"score": score, "family": family, "columns": columns, "model": estimator, "row": row}
    test_predictions = best["model"].predict(features_array[test_indices][:, best["columns"]])
    _, test_metrics = selection_score(labels[test_indices], test_predictions)
    bundle = {
        "model": best["model"],
        "columns": best["columns"],
        "feature_names": np.asarray(feature_names),
        "family": best["family"],
        "validation": best["row"],
    }
    joblib.dump(bundle, arguments.output / "estimator.joblib")
    report = {
        "seed": SEED,
        "payload_start": payload_start,
        "temporal_blocks": 4,
        "feature_dimension": len(feature_names),
        "selected": best["row"],
        "test": test_metrics,
        "split_geometry_counts": {name: len(values) for name, values in split.items()},
    }
    (arguments.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {arguments.output}")


if __name__ == "__main__":
    main()
