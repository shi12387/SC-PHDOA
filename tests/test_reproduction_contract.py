"""Contracts for the final paper reproduction protocol."""

from __future__ import annotations

import json
import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.protocol import load_protocol, protocol_fingerprint


class ReproductionContractTest(unittest.TestCase):
    def test_final_estimator_protocol(self) -> None:
        protocol = load_protocol()
        self.assertEqual(protocol["als_iterations"], 1)
        self.assertFalse(protocol["final_iq_refiner"])
        self.assertEqual(
            protocol["guard_non_residual_branch"], "nominal_manifold"
        )
        self.assertEqual(protocol["context_subsampling_repeats"], 10)

    def test_all_dataset_names_are_unique(self) -> None:
        names = list(load_protocol()["datasets"].values())
        self.assertEqual(len(names), len(set(names)))

    def test_protocol_fingerprint_is_stable_sha256(self) -> None:
        fingerprint = protocol_fingerprint()
        self.assertEqual(len(fingerprint), 64)
        int(fingerprint, 16)

    def test_matlab_generators_cover_declared_datasets(self) -> None:
        generated = {
            "source_count_training.mat",
            "session_calibration_training.mat",
            "external_session_test_L1024_states300_scenes18_N5400_P256.mat",
            "mild_ood_array_impairments.mat",
            "strong_ood_array_impairments.mat",
            "close_source_spatial_only.mat",
            "close_source_joint_hard.mat",
            "drift_gain_phase.mat",
            "drift_coupling.mat",
            "drift_position.mat",
            "drift_compound.mat",
        }
        self.assertEqual(set(load_protocol()["datasets"].values()), generated)

    def test_guard_threshold_is_a_frozen_artifact(self) -> None:
        threshold_path = (
            ROOT / "checkpoints" / "guard_routing" / "threshold.json"
        )
        payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(payload["threshold"], 0.00012714504506827033)
        self.assertEqual(payload["residual_route_when"], "score > threshold")

    def test_experiment_checkpoint_manifest(self) -> None:
        manifest = json.loads(
            (ROOT / "checkpoints" / "EXPERIMENT_CHECKPOINTS.json").read_text(encoding="utf-8")
        )
        for relative, expected in manifest["files"].items():
            path = ROOT / "checkpoints" / relative
            self.assertTrue(path.is_file(), relative)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest.lower(), expected.lower())


if __name__ == "__main__":
    unittest.main()
