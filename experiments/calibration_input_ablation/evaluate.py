"""Evaluate analytic-only, IQ-only, and full calibration-input variants."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sc_phdoa.calibration import CalibrationInputAblationNetwork, SessionCalibrationNetwork
from sc_phdoa.evaluation import (
    evaluate_pipeline,
    paired_session_bootstrap,
    save_standard_reports,
    summarize,
    write_csv,
)
from sc_phdoa.inference import SCPHDOAInferencePipeline
from sc_phdoa.protocol import dataset_path, output_path

MAT_PATH = dataset_path("external_test")
TRAINED_CHECKPOINT_DIRECTORY = output_path("calibration_input_ablation")
BUNDLED_CHECKPOINT_DIRECTORY = ROOT / "checkpoints" / "calibration_input_ablation"
OUTPUT_DIRECTORY = output_path("calibration_input_ablation_evaluation")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_variant(variant: str) -> torch.nn.Module:
    trained = TRAINED_CHECKPOINT_DIRECTORY / variant / "network.pt"
    checkpoint_path = trained if trained.is_file() else BUNDLED_CHECKPOINT_DIRECTORY / variant / "network.pt"
    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=True
    )
    model = (
        SessionCalibrationNetwork()
        if variant == "full"
        else CalibrationInputAblationNetwork(variant)
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> None:
    pipeline = SCPHDOAInferencePipeline.from_files(
        MAT_PATH, device=DEVICE, validate_identity=True
    )
    all_rows = []
    for variant in ("analytic_only", "iq_only", "full"):
        print(f"Evaluating calibration-input variant: {variant}")
        pipeline.model.components.session_calibrator = load_variant(variant)
        variant_output = OUTPUT_DIRECTORY / variant
        rows = evaluate_pipeline(
            pipeline,
            MAT_PATH,
            variant_output,
            calibration_route="session",
            reuse_preprocessing=True,
        )
        for row in rows:
            row["variant"] = variant
            row["method"] = variant
        save_standard_reports(rows, variant_output)
        all_rows.extend(rows)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIRECTORY / "scene_details.csv", all_rows)
    write_csv(
        OUTPUT_DIRECTORY / "variant_summary.csv",
        summarize(all_rows, ("variant",)),
    )
    write_csv(
        OUTPUT_DIRECTORY / "condition_summary.csv",
        summarize(all_rows, ("variant", "case")),
    )
    write_csv(
        OUTPUT_DIRECTORY / "full_vs_analytic_paired_bootstrap.csv",
        paired_session_bootstrap(all_rows, "analytic_only", "full"),
    )


if __name__ == "__main__":
    main()
