"""Basic contracts for the released model architecture."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.calibration import CalibrationInputAblationNetwork, SessionCalibrationNetwork
from sc_phdoa.baselines.neural_set import NeuralSetBaseline
from sc_phdoa.guard_features import build_guard_features


class ArchitectureTest(unittest.TestCase):
    def test_session_calibration_shapes_and_parameter_count(self) -> None:
        model = SessionCalibrationNetwork().eval()
        analytic = torch.zeros(2, 8, 3, 26)
        mask = torch.ones(2, 8, 3, dtype=torch.bool)
        iq = torch.zeros(2, 8, 3, 2, 7, 256)
        with torch.inference_mode():
            output = model(analytic, mask, iq)
        self.assertEqual(tuple(output["final"].shape), (2, 50))
        self.assertEqual(tuple(output["gates"].shape), (2, 3))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 93533)

    def test_calibration_input_ablation_shapes(self) -> None:
        analytic = torch.zeros(1, 8, 3, 26)
        mask = torch.ones(1, 8, 3, dtype=torch.bool)
        iq = torch.zeros(1, 8, 3, 2, 7, 256)
        for variant in CalibrationInputAblationNetwork.VARIANTS:
            model = CalibrationInputAblationNetwork(variant).eval()
            with torch.inference_mode():
                output = model(analytic, mask, iq)
            self.assertEqual(tuple(output["final"].shape), (1, 50))

    def test_guard_feature_dimensions(self) -> None:
        token = torch.zeros(43).numpy()
        summary, features = build_guard_features(
            torch.zeros(50).numpy(), torch.zeros(3).numpy(), 0.2, 0.1, [token]
        )
        self.assertEqual(summary.shape, (7,))
        self.assertEqual(features.shape, (190,))

    def test_controlled_setnet_shapes(self) -> None:
        for support_count in (8, 12):
            scenes = support_count + 1
            for representation in ("cov", "iq"):
                model = NeuralSetBaseline(representation).eval()
                shape = ((1, scenes, 2, 7, 7) if representation == "cov" else (1, scenes, 2, 7, 1024))
                with torch.inference_mode():
                    output = model(torch.zeros(shape), torch.zeros(1, scenes, 1))
                self.assertEqual(tuple(output["angles"].shape), (1, 3, 2))
                self.assertEqual(tuple(output["exist_logits"].shape), (1, 3))
                self.assertEqual(tuple(output["count_logits"].shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
