"""Neural encoders used by the final session-calibration network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ArmAwareIQEncoder(nn.Module):
    """Encode aligned complex IQ while respecting the two L-array arms.

    Input shape is ``[batch, scenes, sources, iq, sensors, time]``. The
    reference sensor is included in both arm views, matching the locked model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.arm_front = nn.Sequential(
            nn.Conv1d(8, 24, 7, stride=2, padding=3),
            nn.GroupNorm(6, 24),
            nn.GELU(),
        )
        self.temporal = nn.ModuleList(
            nn.Conv1d(24, 16, kernel, stride=2, padding=kernel // 2)
            for kernel in (3, 9, 21)
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(48, 48, 5, stride=2, padding=2),
            nn.GroupNorm(8, 48),
            nn.GELU(),
        )
        self.pool = nn.Sequential(
            nn.Linear(96, 48),
            nn.GELU(),
            nn.LayerNorm(48),
        )
        self.cross_arm = nn.Sequential(
            nn.Linear(192, 96),
            nn.GELU(),
            nn.LayerNorm(96),
            nn.Linear(96, 64),
            nn.GELU(),
        )

    def encode_arm(self, value: torch.Tensor) -> torch.Tensor:
        value = self.arm_front(value)
        scales = [F.gelu(layer(value)) for layer in self.temporal]
        value = self.fuse(torch.cat(scales, dim=1))
        mean = value.mean(dim=-1)
        std = value.var(dim=-1, unbiased=False).add(1e-6).sqrt()
        return self.pool(torch.cat([mean, std], dim=1))

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        batch, scenes, sources, _, _, length = iq.shape
        flat = iq.reshape(batch * scenes * sources, 2, 7, length)
        horizontal = flat[:, :, [0, 1, 2, 3]].reshape(-1, 8, length)
        vertical = flat[:, :, [0, 4, 5, 6]].reshape(-1, 8, length)
        horizontal = self.encode_arm(horizontal)
        vertical = self.encode_arm(vertical)
        combined = torch.cat(
            [horizontal, vertical, torch.abs(horizontal - vertical), horizontal * vertical],
            dim=1,
        )
        return self.cross_arm(combined).reshape(batch, scenes, sources, 64)
