"""Regression checks for the paper EM-SBL input and resume cache."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.baselines.sbl import normalized_scene_covariance


def load_evaluator():
    path = ROOT / "experiments" / "em_sbl" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("em_sbl_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EmSblTest(unittest.TestCase):
    def test_covariance_matches_paper_cache_preprocessing(self) -> None:
        rng = np.random.default_rng(2026)
        snapshots = (
            rng.standard_normal((7, 128)) + 1j * rng.standard_normal((7, 128))
        ).astype(np.complex64)
        centered = snapshots - snapshots.mean(axis=1, keepdims=True)
        covariance = centered @ centered.conj().T / snapshots.shape[1]
        power = max(float(np.trace(covariance).real / 7.0), 1e-10)
        normalized = covariance / power
        expected = normalized.real.astype(np.float32) + 1j * normalized.imag.astype(np.float32)
        np.testing.assert_array_equal(normalized_scene_covariance(snapshots), expected)

    def test_state_cache_reuses_only_matching_signature_and_counts(self) -> None:
        evaluator = load_evaluator()
        scenes = np.asarray([12, 13], np.int64)
        cache = {
            "signature": np.asarray("paper-protocol"),
            "state_index": np.asarray(0, np.int64),
            "state_id": np.asarray(1, np.int64),
            "scenes": scenes,
            "counts": np.asarray([1, 2], np.int64),
            "angles": np.zeros((2, 3, 2), np.float32),
            "runtime_ms": np.asarray(8.0),
        }
        test_root = Path(os.environ.get("SC_PHDOA_TEST_TMP", ROOT / "outputs"))
        with tempfile.TemporaryDirectory(dir=test_root) as directory:
            path = Path(directory) / "state_0000.npz"
            evaluator._save_state_cache(path, cache)
            self.assertIsNotNone(evaluator._load_state_cache(
                path, "paper-protocol", 0, 1, scenes, {(0, 12): 1, (0, 13): 2}
            ))
            self.assertIsNone(evaluator._load_state_cache(
                path, "changed-protocol", 0, 1, scenes, {}
            ))
            self.assertIsNone(evaluator._load_state_cache(
                path, "paper-protocol", 0, 1, scenes, {(0, 12): 2, (0, 13): 2}
            ))


if __name__ == "__main__":
    unittest.main()
