"""Cross-scene session calibration network."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn

from .config import DEFAULT_CONFIG, ModelConfig
from .encoders import ArmAwareIQEncoder


class CalibrationOutput(TypedDict):
    raw: torch.Tensor
    final: torch.Tensor
    gates: torch.Tensor


class SessionCalibrationNetwork(nn.Module):
    """Estimate one shared 50-D calibration state from context scenes.

    The three scalar gates correspond to gain/phase, sensor-position, and
    mutual-coupling components. Attribute names intentionally match the locked
    training implementation so its state dictionary loads without conversion.
    """

    def __init__(self, output_dim: int = DEFAULT_CONFIG.calibration_dim) -> None:
        super().__init__()
        if output_dim < DEFAULT_CONFIG.gain_dim + DEFAULT_CONFIG.position_dim:
            raise ValueError("output_dim is too small for the fixed calibration heads")
        hidden = 48
        self.iq_encoder = ArmAwareIQEncoder()
        self.analytic_encoder = nn.Sequential(
            nn.Linear(DEFAULT_CONFIG.analytic_feature_dim, 48),
            nn.GELU(),
            nn.LayerNorm(48),
            nn.Linear(48, 32),
            nn.GELU(),
        )
        self.source_fusion = nn.Sequential(
            nn.Linear(96, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, hidden),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            hidden,
            4,
            96,
            0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.scene_encoder = nn.TransformerEncoder(layer, 1)
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.GELU(),
        )
        self.gain_head = nn.Linear(64, 12)
        self.position_head = nn.Linear(64, 12)
        self.coupling_head = nn.Linear(64, output_dim - 24)
        self.gate_head = nn.Linear(64, 3)

    def forward(
        self,
        analytic_features: torch.Tensor,
        source_mask: torch.Tensor,
        aligned_iq: torch.Tensor,
    ) -> CalibrationOutput:
        iq_value = self.iq_encoder(aligned_iq)
        analytic = self.analytic_encoder(analytic_features)
        source = self.source_fusion(torch.cat([iq_value, analytic], dim=-1))

        weights = source_mask.to(source.dtype).unsqueeze(-1)
        scenes = (source * weights).sum(2) / weights.sum(2).clamp_min(1.0)
        encoded = self.scene_encoder(scenes).mean(dim=1)
        shared = self.shared(encoded)

        gain = torch.tanh(self.gain_head(shared))
        position = torch.tanh(self.position_head(shared))
        coupling = torch.tanh(self.coupling_head(shared))
        gates = torch.sigmoid(self.gate_head(shared))
        raw = torch.cat([gain, position, coupling], dim=1)
        final = torch.cat(
            [
                gain * gates[:, 0:1],
                position * gates[:, 1:2],
                coupling * gates[:, 2:3],
            ],
            dim=1,
        )
        return {"raw": raw, "final": final, "gates": gates}


class CalibrationInputAblationNetwork(nn.Module):
    """Analytic-only, IQ-only, or full calibration-input ablation."""

    VARIANTS = ("analytic_only", "iq_only", "full")

    def __init__(self, variant: str, output_dim: int = 50) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown calibration input variant: {variant}")
        self.variant = variant
        hidden = 48
        if variant in ("iq_only", "full"):
            self.iq_encoder = ArmAwareIQEncoder()
        if variant in ("analytic_only", "full"):
            self.analytic_encoder = nn.Sequential(
                nn.Linear(26, 48), nn.GELU(), nn.LayerNorm(48),
                nn.Linear(48, 32), nn.GELU(),
            )
        input_dim = {"analytic_only": 32, "iq_only": 64, "full": 96}[variant]
        projection = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, hidden), nn.GELU(),
        )
        if variant == "full":
            self.source_fusion = projection
        else:
            self.source_projection = projection
        layer = nn.TransformerEncoderLayer(
            hidden, 4, 96, 0.10, activation="gelu", batch_first=True, norm_first=True
        )
        self.scene_encoder = nn.TransformerEncoder(layer, 1)
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 64), nn.GELU()
        )
        self.gain_head = nn.Linear(64, 12)
        self.position_head = nn.Linear(64, 12)
        self.coupling_head = nn.Linear(64, output_dim - 24)
        self.gate_head = nn.Linear(64, 3)

    def forward(
        self,
        analytic_features: torch.Tensor,
        source_mask: torch.Tensor,
        aligned_iq: torch.Tensor,
    ) -> CalibrationOutput:
        if self.variant == "analytic_only":
            source = self.source_projection(self.analytic_encoder(analytic_features))
        elif self.variant == "iq_only":
            source = self.source_projection(self.iq_encoder(aligned_iq))
        else:
            source = self.source_fusion(torch.cat([
                self.iq_encoder(aligned_iq), self.analytic_encoder(analytic_features)
            ], dim=-1))
        weights = source_mask.to(source.dtype).unsqueeze(-1)
        scenes = (source * weights).sum(2) / weights.sum(2).clamp_min(1.0)
        shared = self.shared(self.scene_encoder(scenes).mean(dim=1))
        gain = torch.tanh(self.gain_head(shared))
        position = torch.tanh(self.position_head(shared))
        coupling = torch.tanh(self.coupling_head(shared))
        gates = torch.sigmoid(self.gate_head(shared))
        raw = torch.cat([gain, position, coupling], dim=1)
        final = torch.cat([
            gain * gates[:, 0:1],
            position * gates[:, 1:2],
            coupling * gates[:, 2:3],
        ], dim=1)
        return {"raw": raw, "final": final, "gates": gates}
