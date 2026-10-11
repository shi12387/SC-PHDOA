"""Run, resume, export, and verify the final paper experiment suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "overall": ROOT / "experiments" / "overall_comparison" / "evaluate.py",
    "classical": ROOT / "experiments" / "classical_baselines" / "evaluate.py",
    "em_sbl": ROOT / "experiments" / "em_sbl" / "evaluate.py",
    "observation_budget": ROOT / "experiments" / "observation_budget" / "evaluate.py",
    "calibration": ROOT / "experiments" / "calibration_manifold" / "evaluate.py",
    "oracle": ROOT / "experiments" / "oracle_calibration" / "evaluate.py",
    "oracle_k": ROOT / "experiments" / "oracle_source_count" / "evaluate.py",
    "context": ROOT / "experiments" / "context_sensitivity" / "evaluate.py",
    "ood": ROOT / "experiments" / "ood_generalization" / "evaluate.py",
    "close_source": ROOT / "experiments" / "close_source_resolution" / "evaluate.py",
    "drift": ROOT / "experiments" / "calibration_drift" / "evaluate.py",
    "shuffled": ROOT / "experiments" / "shuffled_session" / "evaluate.py",
    "input_ablation": ROOT / "experiments" / "calibration_input_ablation" / "evaluate.py",
    "runtime": ROOT / "experiments" / "runtime_benchmark" / "benchmark.py",
    "als_depth": ROOT / "experiments" / "als_depth_validation" / "evaluate.py",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final paper experiments")
    parser.add_argument(
        "--experiments", nargs="+", choices=tuple(EXPERIMENTS),
        default=list(EXPERIMENTS), help="Experiments to run in order.",
    )
    parser.add_argument("--skip-environment-check", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def run_step(name: str, script: Path) -> dict:
    print(f"\n===== {name}: {script.relative_to(ROOT)} =====", flush=True)
    started = time.perf_counter()
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT, check=False)
    return {
        "name": name,
        "script": str(script.relative_to(ROOT)),
        "return_code": result.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "ok": result.returncode == 0,
    }


def save_status(steps: list[dict]) -> None:
    destination = ROOT / "outputs" / "reproduction" / "run_status.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"steps": steps}, indent=2), encoding="utf-8")
    temporary.replace(destination)


def main() -> None:
    arguments = parse_arguments()
    if arguments.list:
        for name, script in EXPERIMENTS.items():
            print(f"{name:14s} {script.relative_to(ROOT)}")
        return

    steps: list[dict] = []
    if not arguments.skip_environment_check:
        step = run_step("environment", ROOT / "scripts" / "validate_environment.py")
        steps.append(step)
        save_status(steps)
        if not step["ok"]:
            raise SystemExit(step["return_code"])

    for name in arguments.experiments:
        step = run_step(name, EXPERIMENTS[name])
        steps.append(step)
        save_status(steps)
        if not step["ok"] and not arguments.continue_on_error:
            raise SystemExit(step["return_code"])

    if {"overall", "classical", "em_sbl", "observation_budget"}.issubset(arguments.experiments):
        step = run_step("assemble", ROOT / "scripts" / "assemble_paper_results.py")
        steps.append(step)
        save_status(steps)
        if not step["ok"] and not arguments.continue_on_error:
            raise SystemExit(step["return_code"])

    if not arguments.skip_export:
        step = run_step("export", ROOT / "scripts" / "export_paper_artifacts.py")
        steps.append(step)
        save_status(steps)
    if not arguments.skip_verification:
        step = run_step("verification", ROOT / "scripts" / "verify_reproduction.py")
        steps.append(step)
        save_status(steps)

    failed = [step for step in steps if not step["ok"]]
    print(f"\nCompleted {len(steps)} steps; failures={len(failed)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
