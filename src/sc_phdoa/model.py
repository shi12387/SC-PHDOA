"""Learned components used by the physical SC-PHDOA pipeline."""

from __future__ import annotations

import numpy as np
import torch

from .checkpoints import CheckpointPaths, LoadedComponents, load_components


class SCPHDOAModel:
    def __init__(self, components: LoadedComponents, device: str) -> None:
        self.components = components
        self.device = torch.device(device)

    @classmethod
    def from_checkpoints(
        cls, paths: CheckpointPaths | None = None, device: str | None = None
    ) -> "SCPHDOAModel":
        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(load_components(paths, selected), selected)

    def predict_source_count(
        self,
        snapshots: np.ndarray,
        payload_start: int,
        preamble: np.ndarray,
        maximum_delay: int,
    ):
        return self.components.source_count.predict(
            snapshots, payload_start, preamble, maximum_delay
        )

    @torch.inference_mode()
    def estimate_session_calibration(
        self,
        analytic_features: np.ndarray,
        source_mask: np.ndarray,
        aligned_iq: np.ndarray,
    ) -> dict[str, np.ndarray]:
        output = self.components.session_calibrator(
            torch.as_tensor(analytic_features, device=self.device),
            torch.as_tensor(source_mask, device=self.device),
            torch.as_tensor(aligned_iq, device=self.device),
        )
        return {key: value.detach().cpu().numpy() for key, value in output.items()}
