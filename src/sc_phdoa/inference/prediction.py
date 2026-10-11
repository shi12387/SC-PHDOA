"""Typed inference outputs and serialization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class TargetPrediction:
    scene_index: int
    source_count: int
    doas_deg: np.ndarray
    als_diagnostics: np.ndarray

    def summary(self) -> Dict[str, Any]:
        return {
            "scene_index": int(self.scene_index),
            "source_count": int(self.source_count),
            "doas_deg": np.asarray(self.doas_deg).tolist(),
            "als_diagnostics": np.asarray(self.als_diagnostics).tolist(),
        }


@dataclass(frozen=True)
class SessionPrediction:
    state_index: int
    state_id: int
    calibration_route: str
    all_scene_source_counts: np.ndarray
    targets: List[TargetPrediction]
    arrays: Dict[str, np.ndarray]

    def summary(self) -> Dict[str, Any]:
        return {
            "state_index": int(self.state_index),
            "state_id": int(self.state_id),
            "calibration_route": self.calibration_route,
            "all_scene_source_counts": self.all_scene_source_counts.astype(int).tolist(),
            "targets": [target.summary() for target in self.targets],
        }

    def save(self, output_stem: Path) -> None:
        output_stem = Path(output_stem)
        output_stem.parent.mkdir(parents=True, exist_ok=True)
        with output_stem.with_suffix(".json").open("w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2)
        np.savez_compressed(output_stem.with_suffix(".npz"), **self.arrays)
