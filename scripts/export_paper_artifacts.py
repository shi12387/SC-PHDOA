"""Export publication-ready summary tables and Figures 4--7."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DESTINATION = OUTPUTS / "paper_artifacts"
TABLES = DESTINATION / "tables"
FIGURES = DESTINATION / "figures"

LABELS = {
    "mdl_music": "MDL-MUSIC",
    "music": "MUSIC",
    "esprit": "ESPRIT",
    "em_sbl": "2-D EM-SBL",
    "cov_setnet_support8": "Cov-SetNet",
    "iq_setnet_support8": "IQ-SetNet",
    "nominal_als": "Nominal ALS",
    "session_calibrated_als": "Session-Calibrated ALS",
    "guard_routed_als": "SC-PHDOA",
    "sc_phdoa": "SC-PHDOA",
    "oracle_calibrated_als": "Oracle-Calibrated ALS",
}


def read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment output: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def figure_overall(rows: list[dict[str, str]]) -> None:
    order = [
        "mdl_music", "music", "esprit", "em_sbl",
        "cov_setnet_support8", "iq_setnet_support8",
        "nominal_als", "sc_phdoa",
    ]
    selected = {row["method"]: row for row in rows}
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.2))
    labels = [LABELS[name] for name in order]
    axes[0].bar(labels, [float(selected[name]["mean_ospa"]) for name in order])
    axes[0].set_ylabel("OSPA (deg)")
    axes[1].bar(labels, [100 * float(selected[name]["e2e_3deg"]) for name in order])
    axes[1].set_ylabel("E2E@3 deg (%)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35, labelsize=7)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save_figure(figure, "figure_4_pipeline_comparison")


def figure_robustness() -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    snr_rows = read(OUTPUTS / "assembled_paper_results" / "E06_snr_robustness.csv")
    count_rows = read(OUTPUTS / "assembled_paper_results" / "E07_source_count_breakdown.csv")
    for method in ("music", "cov_setnet_support8", "nominal_als", "sc_phdoa"):
        snr = [row for row in snr_rows if row["method"] == method]
        snr.sort(key=lambda row: float(row["snr"]))
        axes[0].plot(
            [float(row["snr"]) for row in snr],
            [float(row["mean_scene_correct_k_rmse"]) for row in snr],
            marker="o", label=LABELS[method],
        )
        count = [row for row in count_rows if row["method"] == method]
        count.sort(key=lambda row: int(float(row["K_true"])))
        axes[1].plot(
            [int(float(row["K_true"])) for row in count],
            [float(row["mean_scene_correct_k_rmse"]) for row in count],
            marker="o", label=LABELS[method],
        )
    axes[0].set(xlabel="SNR (dB)", ylabel="Correct-K RMSE (deg)")
    axes[1].set(xlabel="Source count K", ylabel="Correct-K RMSE (deg)")
    axes[1].set_xticks([1, 2, 3])
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, "figure_5_robustness")


def figure_close_source(rows: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for axis, mode in zip(axes, ("spatial_only", "joint_hard")):
        for method in ("nominal_als", "sc_phdoa"):
            for source_count in (2, 3):
                selected = [
                    row for row in rows
                    if row["mode"] == mode and row["method"] == method
                    and int(float(row["K_true"])) == source_count
                ]
                selected.sort(key=lambda row: float(row["target_separation_deg"]))
                axis.plot(
                    [float(row["target_separation_deg"]) for row in selected],
                    [100 * float(row["resolution_success"]) for row in selected],
                    marker="o", label=f"{LABELS[method]}, K={source_count}",
                )
        axis.set_title(mode.replace("_", " ").title())
        axis.set_xlabel("Minimum separation (deg)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Resolution success (%)")
    axes[1].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    save_figure(figure, "figure_6_close_source_resolution")


def figure_drift(rows: list[dict[str, str]]) -> None:
    components = ("gain_phase", "coupling", "position", "compound")
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    for axis, component in zip(axes.ravel(), components):
        for method in ("nominal_als", "session_calibrated_als", "guard_routed_als"):
            selected = [
                row for row in rows
                if row["drift_component"] == component and row["method"] == method
            ]
            selected.sort(key=lambda row: float(row["drift_lambda"]))
            axis.plot(
                [float(row["drift_lambda"]) for row in selected],
                [float(row["mean_ospa"]) for row in selected],
                marker="o", label=LABELS[method],
            )
        axis.set_title(component.replace("_", "/").title())
        axis.set_xlabel("Drift strength")
        axis.set_ylabel("OSPA (deg)")
        axis.grid(alpha=0.25)
    axes[0, 1].legend(frameon=False, fontsize=7)
    figure.tight_layout()
    save_figure(figure, "figure_7_context_target_mismatch")


def main() -> None:
    sources = {
        "overall": OUTPUTS / "assembled_paper_results" / "E01_overall_comparison.csv",
        "snr": OUTPUTS / "assembled_paper_results" / "E06_snr_robustness.csv",
        "source_count": OUTPUTS / "assembled_paper_results" / "E07_source_count_breakdown.csv",
        "array_condition": OUTPUTS / "assembled_paper_results" / "E09_array_condition_breakdown.csv",
        "observation_budget": OUTPUTS / "observation_budget" / "method_summary.csv",
        "observation_budget_ci": OUTPUTS / "observation_budget" / "support12_minus_support8_bootstrap.csv",
        "calibration_manifold": OUTPUTS / "calibration_manifold" / "condition_summary.csv",
        "oracle": OUTPUTS / "oracle_calibration" / "method_summary.csv",
        "oracle_k": OUTPUTS / "oracle_source_count" / "method_summary.csv",
        "context": OUTPUTS / "context_sensitivity" / "context_summary.csv",
        "ood": OUTPUTS / "ood_generalization" / "distribution_summary.csv",
        "ood_manifold_reduction": OUTPUTS / "ood_generalization" / "manifold_relative_reduction.csv",
        "close_source": OUTPUTS / "close_source_resolution" / "resolution_summary.csv",
        "drift": OUTPUTS / "calibration_drift" / "drift_summary.csv",
        "shuffled_session": OUTPUTS / "shuffled_session" / "method_summary.csv",
        "calibration_input_ablation": OUTPUTS / "calibration_input_ablation_evaluation" / "variant_summary.csv",
        "runtime_modules": OUTPUTS / "runtime_benchmark" / "module_latency_summary.csv",
        "als_depth_validation": OUTPUTS / "als_depth_validation" / "summary.csv",
    }
    loaded = {name: read(path) for name, path in sources.items()}
    for name, rows in loaded.items():
        write(TABLES / f"{name}.csv", rows)
    figure_overall(loaded["overall"])
    figure_robustness()
    figure_close_source(loaded["close_source"])
    figure_drift(loaded["drift"])
    manifest = {name: str(path.relative_to(ROOT)) for name, path in sources.items()}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Paper artifacts: {DESTINATION}")


if __name__ == "__main__":
    main()
