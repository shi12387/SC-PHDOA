"""Controlled covariance and raw-IQ set-prediction baselines."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class Residual2D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, output_channels), nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, output_channels),
        )
        self.skip = nn.Identity() if input_channels == output_channels and stride == 1 else nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.main(value) + self.skip(value))


class CovarianceSceneEncoder(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
            Residual2D(32, 32), Residual2D(32, 64, stride=2), Residual2D(64, 96, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.project = nn.Sequential(nn.Linear(97, token_dim), nn.GELU(), nn.LayerNorm(token_dim))

    def forward(self, signal: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
        batch, scenes = signal.shape[:2]
        value = self.network(signal.reshape(-1, 2, 7, 7)).flatten(1)
        return self.project(torch.cat([value, power.reshape(-1, 1)], dim=1)).reshape(batch, scenes, -1)


class TemporalResidual(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False), nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False), nn.GroupNorm(8, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(value + self.network(value))


class ArmTemporalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(8, 32, 9, stride=4, padding=4, bias=False), nn.GroupNorm(8, 32), nn.GELU())
        self.multiscale = nn.ModuleList([nn.Conv1d(32, 24, kernel, stride=2, padding=kernel // 2, bias=False) for kernel in (3, 9, 21)])
        self.fuse = nn.Sequential(
            nn.Conv1d(72, 64, 1, bias=False), nn.GroupNorm(8, 64), nn.GELU(),
            TemporalResidual(64, 1), TemporalResidual(64, 2), TemporalResidual(64, 4),
        )
        self.pool = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.LayerNorm(64))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.stem(value)
        value = torch.cat([torch.nn.functional.gelu(layer(value)) for layer in self.multiscale], dim=1)
        value = self.fuse(value)
        return self.pool(torch.cat([value.mean(dim=-1), value.var(dim=-1, unbiased=False).add(1e-6).sqrt()], dim=1))


class IQSceneEncoder(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self.arm = ArmTemporalEncoder()
        self.cross_arm = nn.Sequential(nn.Linear(257, 384), nn.GELU(), nn.LayerNorm(384), nn.Linear(384, token_dim), nn.GELU(), nn.LayerNorm(token_dim))

    def forward(self, signal: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
        batch, scenes, _, _, length = signal.shape
        flat = signal.reshape(batch * scenes, 2, 7, length)
        x_value = self.arm(flat[:, :, [0, 1, 2, 3]].reshape(-1, 8, length))
        z_value = self.arm(flat[:, :, [0, 4, 5, 6]].reshape(-1, 8, length))
        combined = torch.cat([x_value, z_value, torch.abs(x_value - z_value), x_value * z_value, power.reshape(-1, 1)], dim=1)
        return self.cross_arm(combined).reshape(batch, scenes, -1)


class NeuralSetBaseline(nn.Module):
    def __init__(self, representation: str, token_dim: int = 128, encoder_layers: int = 2, decoder_layers: int = 2, attention_heads: int = 4):
        super().__init__()
        self.scene_encoder = CovarianceSceneEncoder(token_dim) if representation == "cov" else IQSceneEncoder(token_dim)
        self.representation = representation
        self.role_embedding = nn.Parameter(torch.zeros(2, token_dim))
        nn.init.normal_(self.role_embedding, std=0.02)
        encoder = nn.TransformerEncoderLayer(token_dim, attention_heads, token_dim * 2, 0.10, activation="gelu", batch_first=True, norm_first=True)
        decoder = nn.TransformerDecoderLayer(token_dim, attention_heads, token_dim * 2, 0.10, activation="gelu", batch_first=True, norm_first=True)
        self.context_encoder = nn.TransformerEncoder(encoder, encoder_layers)
        self.query_decoder = nn.TransformerDecoder(decoder, decoder_layers)
        self.queries = nn.Parameter(torch.randn(3, token_dim) * 0.02)
        self.query_head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, 64), nn.GELU(), nn.Linear(64, 3))
        self.count_head = nn.Sequential(nn.LayerNorm(token_dim * 2), nn.Linear(token_dim * 2, 96), nn.GELU(), nn.Dropout(0.10), nn.Linear(96, 3))

    def forward(self, signal: torch.Tensor, power: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.scene_encoder(signal, power)
        roles = torch.zeros(tokens.shape[:2], dtype=torch.long, device=tokens.device)
        roles[:, -1] = 1
        encoded = self.context_encoder(tokens + self.role_embedding[roles])
        decoded = self.query_decoder(self.queries[None].expand(len(tokens), -1, -1), encoded)
        raw = self.query_head(decoded)
        angles = torch.stack([60.0 * torch.tanh(raw[..., 0]), 5.0 + 75.0 * torch.sigmoid(raw[..., 1])], dim=-1)
        context = encoded[:, :-1].mean(dim=1)
        count_logits = self.count_head(torch.cat([encoded[:, -1], context], dim=1))
        return {"angles": angles, "exist_logits": raw[..., 2], "count_logits": count_logits}


def load_setnet_checkpoint(path: Path, representation: str, device: str) -> tuple[NeuralSetBaseline, dict]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("representation") != representation or checkpoint.get("phase") != "selection":
        raise ValueError(f"Incompatible {representation} selection checkpoint: {path}")
    model = NeuralSetBaseline(
        representation,
        int(checkpoint["token_dim"]),
        int(checkpoint["encoder_layers"]),
        int(checkpoint["decoder_layers"]),
        int(checkpoint["attention_heads"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint
