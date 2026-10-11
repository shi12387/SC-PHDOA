# Reproducibility Status

This file defines the exact scope of the reviewer-facing package. It is a
release checklist, not an experimental claim.

## Included and wired into the paper runner

- Deterministic generation of training, external-ID, OOD, close-source, and
  context-target mismatch MAT datasets.
- Source-count, session-calibration, and guard residual/router training.
- Frozen predicted-K inference with one ALS iteration, no final IQ refiner,
  and nominal-manifold fallback for the non-residual guard route.
- Overall Nominal/Session-Calibrated/SC-PHDOA comparison, including
  session-level bootstrap confidence intervals and SNR/source-count/
  impairment breakdowns.
- Oracle-calibration, context-scene sensitivity, ID/mild/strong OOD,
  close-source resolution, and context-target mismatch experiments.
- Direct session wall-clock latency, amortized target latency, module timing,
  neural parameter count, and peak CUDA memory.
- Session-level resumability, protocol fingerprints, reference-metric parity
  checks, and generation of the paper-facing tables and figures.
- MDL-MUSIC, MUSIC, L-array ESPRIT (including explicit MUSIC-fallback count),
  two-dimensional EM-SBL, Cov-SetNet, and IQ-SetNet baselines.
- Eight-versus-twelve-support controlled-baseline retraining and evaluation.
- Target-only Oracle-K, same-condition shuffled-session control, physical
  calibration-component metrics, and validation-only ALS-depth selection.

## Included training analyses

- Calibration-input ablation (analytic-only, IQ-only, and full input).

The package includes both frozen validation-selected checkpoints and clean
training entry points. Adapted published neural methods are deliberately not
claimed as native baselines; the submitted comparison uses the faithful
classical/SBL methods plus explicitly labelled controlled SetNets.

## Release gate

A release is ready only when all of the following hold on the experiment
machine:

1. `python -m unittest discover -s tests -v` passes.
2. `python scripts/validate_environment.py` passes.
3. `python scripts/run_paper_experiments.py` completes without failures.
4. `outputs/reproduction/verification_report.json` contains `"ok": true`.
5. Every comparison method shown in the submitted paper is either migrated
   into this repository or explicitly distributed as a separate artifact
   with code, checkpoint, and provenance.
