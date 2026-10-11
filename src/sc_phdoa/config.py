"""Fixed dimensions and signal-processing settings used by SC-PHDOA."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    sensors: int = 7
    max_sources: int = 3
    analytic_feature_dim: int = 26
    calibration_dim: int = 50
    gain_dim: int = 12
    position_dim: int = 12
    coupling_dim: int = 26
    calibration_gate_dim: int = 3
    als_iterations: int = 1
    als_ridge: float = 3e-4
    als_blend: float = 0.65
    azimuth_bounds_deg: tuple[float, float] = (-60.0, 60.0)
    elevation_bounds_deg: tuple[float, float] = (5.0, 80.0)
    temporal_blocks: int = 4
    sync_fft_size: int = 1024
    cfo_search_bounds: tuple[float, float] = (-0.012, 0.012)
    nms_delay: int = 10
    nms_cfo: float = 0.0013
    preamble_samples: int = 256
    maximum_delay: int = 128
    payload_start: int = 384
    coarse_grid_step_deg: float = 5.0
    initial_multistart_count: int = 3
    iterative_multistart_count: int = 1
    multistart_nms_deg: float = 8.0
    fit_max_evaluations: int = 65
    residual_relative_tolerance: float = 5e-4
    maximum_steering_condition: float = 3000.0
    gain_scale: float = 0.30
    position_scale_spacing: float = 0.20
    coupling_scale: float = 0.35

    def __post_init__(self) -> None:
        if self.gain_dim + self.position_dim + self.coupling_dim != self.calibration_dim:
            raise ValueError("Calibration component dimensions must sum to calibration_dim")


DEFAULT_CONFIG = ModelConfig()
