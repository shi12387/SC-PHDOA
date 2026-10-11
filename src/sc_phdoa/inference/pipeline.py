"""Session-level SC-PHDOA inference pipeline."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..checkpoints import CheckpointPaths
from ..config import DEFAULT_CONFIG
from ..model import SCPHDOAModel

from .backend import SignalProcessingBackend
from .prediction import SessionPrediction, TargetPrediction


class SCPHDOAInferencePipeline:
    """Run predicted-K SC-PHDOA without accessing target labels.

    One call processes all 18 observations of an external calibration session:
    eight context scenes estimate calibration, four guard scenes select the
    route, and six target scenes produce source counts and 2-D DOAs.
    """

    def __init__(
        self,
        model: SCPHDOAModel,
        backend: SignalProcessingBackend,
    ) -> None:
        self.model = model
        self.backend = backend
        self._preprocessed_sessions: Dict[int, Dict[str, Any]] = {}

    @classmethod
    def from_files(
        cls,
        external_mat: Path,
        checkpoints: CheckpointPaths | None = None,
        device: Optional[str] = None,
        validate_identity: bool = True,
    ) -> "SCPHDOAInferencePipeline":
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        backend = SignalProcessingBackend.load(external_mat, validate_identity)
        model = SCPHDOAModel.from_checkpoints(checkpoints, selected_device)
        return cls(model=model, backend=backend)

    @property
    def state_count(self) -> int:
        return self.backend.state_count

    def _estimate_calibration(
        self,
        frontends: List[Dict[str, Any]],
        context_scenes: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        scene_count = len(frontends)
        maximum_sources = 3
        analytic_dim = 26
        iq_shape = frontends[0]["aligned_iq"].shape[1:]
        features = np.zeros(
            (scene_count, maximum_sources, analytic_dim), np.float32
        )
        source_mask = np.zeros((scene_count, maximum_sources), bool)
        aligned_iq = np.zeros(
            (scene_count, maximum_sources, *iq_shape),
            np.float32,
        )
        for scene, frontend in enumerate(frontends):
            count = int(frontend["count"])
            features[scene, :count] = frontend["analytic"]
            source_mask[scene, :count] = True
            aligned_iq[scene, :count] = frontend["aligned_iq"]
        selected = np.asarray(context_scenes, np.int64)
        output = self.model.estimate_session_calibration(
            features[selected][None],
            source_mask[selected][None],
            aligned_iq[selected][None],
        )
        return {key: value[0] for key, value in output.items()}

    def predict_session(
        self,
        state_index: int,
        context_scenes: np.ndarray | None = None,
        calibration_route: str = "guard",
        snapshots_by_scene: List[np.ndarray] | None = None,
        profile: Dict[str, float] | None = None,
        reuse_preprocessing: bool = False,
        calibration_source_state: int | None = None,
        target_count_override: Dict[int, int] | None = None,
        manifold_override: Dict[str, Any] | None = None,
    ) -> SessionPrediction:
        state_index = int(state_index)
        if state_index < 0 or state_index >= self.state_count:
            raise IndexError(
                f"state_index={state_index} outside [0, {self.state_count - 1}]"
            )

        def measured(name: str, operation):
            if profile is None:
                return operation()
            if self.model.device.type == "cuda":
                torch.cuda.synchronize(self.model.device)
            started = time.perf_counter()
            value = operation()
            if self.model.device.type == "cuda":
                torch.cuda.synchronize(self.model.device)
            profile[name] = profile.get(name, 0.0) + 1000.0 * (
                time.perf_counter() - started
            )
            return value

        def build_frontend(snapshots: np.ndarray, count: int):
            snr = self.backend.estimate_snr(snapshots, count)
            frontend = self.backend.build_source_frontend(snapshots, count, snr)
            return snr, frontend

        cached = self._preprocessed_sessions.get(state_index)
        if reuse_preprocessing and cached is not None:
            snapshots_by_scene = cached["snapshots_by_scene"]
            predicted_counts = cached["predicted_counts"]
            count_probabilities = cached["count_probabilities"]
            snr_estimates = cached["snr_estimates"]
            frontends = cached["frontends"]
        else:
            snapshots_by_scene = (
                self.backend.read_session(state_index)
                if snapshots_by_scene is None
                else snapshots_by_scene
            )
            scene_count = len(snapshots_by_scene)
            predicted_counts = np.zeros(scene_count, np.int64)
            count_probabilities = np.zeros((scene_count, 3), np.float32)
            snr_estimates = np.zeros(scene_count, np.float32)
            frontends: List[Dict[str, Any]] = []
            for scene, snapshots in enumerate(snapshots_by_scene):
                prediction = measured(
                    "source_count",
                    lambda snapshots=snapshots: self.model.predict_source_count(
                        snapshots,
                        self.backend.payload_start,
                        self.backend.preamble,
                        self.backend.maximum_delay,
                    ),
                )
                count = int(prediction.count)
                snr, frontend = measured(
                    "synchronization_frontend",
                    lambda snapshots=snapshots, count=count: build_frontend(
                        snapshots, count
                    ),
                )
                frontends.append(frontend)
                predicted_counts[scene] = count
                count_probabilities[scene] = prediction.probabilities
                snr_estimates[scene] = snr
            if reuse_preprocessing:
                target_snapshots: List[np.ndarray | None] = [None] * scene_count
                for target_scene in self.backend.dataset.target_scenes[state_index]:
                    target_snapshots[int(target_scene)] = snapshots_by_scene[
                        int(target_scene)
                    ]
                self._preprocessed_sessions[state_index] = {
                    "snapshots_by_scene": target_snapshots,
                    "predicted_counts": predicted_counts,
                    "count_probabilities": count_probabilities,
                    "snr_estimates": snr_estimates,
                    "frontends": frontends,
                }

        if target_count_override:
            predicted_counts = predicted_counts.copy()
            frontends = list(frontends)
            for scene, overridden_count in target_count_override.items():
                scene = int(scene)
                overridden_count = int(overridden_count)
                if int(frontends[scene]["count"]) != overridden_count:
                    snr = self.backend.estimate_snr(
                        snapshots_by_scene[scene], overridden_count
                    )
                    frontends[scene] = self.backend.build_source_frontend(
                        snapshots_by_scene[scene], overridden_count, snr
                    )
                    snr_estimates = snr_estimates.copy()
                    snr_estimates[scene] = snr
                predicted_counts[scene] = overridden_count

        calibration_state = (
            state_index
            if calibration_source_state is None
            else int(calibration_source_state)
        )
        if calibration_state != state_index and calibration_route != "session":
            raise ValueError(
                "cross-session calibration is defined only for the "
                "session-calibrated route"
            )
        if calibration_state == state_index:
            calibration_frontends = frontends
        else:
            if calibration_state < 0 or calibration_state >= self.state_count:
                raise IndexError(f"invalid calibration source state: {calibration_state}")
            if target_count_override:
                raise ValueError(
                    "target_count_override and shuffled calibration cannot be combined"
                )
            donor = self._preprocessed_sessions.get(calibration_state)
            if donor is None:
                self.predict_session(
                    calibration_state,
                    calibration_route="nominal",
                    reuse_preprocessing=True,
                )
                donor = self._preprocessed_sessions[calibration_state]
            calibration_frontends = donor["frontends"]

        selected_context = (
            self.backend.dataset.context_scenes[calibration_state]
            if context_scenes is None
            else np.asarray(context_scenes, np.int64)
        )
        calibration = measured(
            "session_calibration",
            lambda: self._estimate_calibration(
                calibration_frontends, selected_context
            ),
        )
        components = self.model.components
        if manifold_override is not None:
            manifold = manifold_override
        elif calibration_route == "guard":
            manifold = measured(
                "guard_routing",
                lambda: self.backend.build_routed_manifold(
                    state_index,
                    frontends,
                    calibration,
                    components.guard_corrector.estimator,
                    components.guard_corrector.dimensions,
                    components.guard_router.threshold,
                ),
            )
        elif calibration_route == "session":
            manifold = self.backend.build_session_manifold(calibration["final"])
        elif calibration_route == "nominal":
            manifold = self.backend.build_nominal_manifold()
        else:
            raise ValueError(f"unknown calibration route: {calibration_route}")

        target_scenes = np.asarray(
            self.backend.dataset.target_scenes[state_index], np.int64
        )
        doa_angles = np.full((len(target_scenes), 3, 2), np.nan, np.float32)
        diagnostics = np.zeros((len(target_scenes), 5), np.float32)
        targets: List[TargetPrediction] = []

        for slot, scene_value in enumerate(target_scenes):
            scene = int(scene_value)
            count = int(predicted_counts[scene])
            angles, diagnostic = measured(
                "target_als",
                lambda scene=scene: self.backend.estimate_target_doas(
                    snapshots_by_scene[scene],
                    frontends[scene],
                    manifold,
                ),
            )
            doa_angles[slot, :count] = angles
            diagnostics[slot] = diagnostic
            targets.append(
                TargetPrediction(
                    scene_index=scene,
                    source_count=count,
                    doas_deg=np.asarray(angles, np.float32),
                    als_diagnostics=np.asarray(diagnostic, np.float32),
                )
            )

        arrays = {
            "state_index": np.asarray(state_index, np.int64),
            "calibration_source_state": np.asarray(
                calibration_state, np.int64
            ),
            "state_id": np.asarray(
                self.backend.dataset.state_ids[state_index], np.int64
            ),
            "target_scenes": target_scenes,
            "predicted_counts": predicted_counts,
            "count_probabilities": count_probabilities,
            "snr_estimates": snr_estimates,
            "calibration_raw": np.asarray(calibration["raw"], np.float32),
            "calibration_final": np.asarray(calibration["final"], np.float32),
            "calibration_gates": np.asarray(calibration["gates"], np.float32),
            "guard_features": np.asarray(manifold["guard_features"], np.float32),
            "route_residual": np.asarray(manifold["use_residual"], bool),
            "initial_calibration": np.asarray(
                manifold["initial_calibration"], np.float32
            ),
            "corrected_calibration": np.asarray(
                manifold["corrected_calibration"], np.float32
            ),
            "target_counts": predicted_counts[target_scenes],
            "doa_angles": doa_angles,
            "als_diagnostics": diagnostics,
            "als_steps": np.asarray(DEFAULT_CONFIG.als_iterations, np.int64),
            "final_iq_refiner": np.asarray(False),
        }
        return SessionPrediction(
            state_index=state_index,
            state_id=int(self.backend.dataset.state_ids[state_index]),
            calibration_route=str(manifold["mode"]),
            all_scene_source_counts=predicted_counts,
            targets=targets,
            arrays=arrays,
        )
