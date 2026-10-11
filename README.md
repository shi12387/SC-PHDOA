# SC-PHDOA

Code and experiment entry points for session-calibrated physical 2-D direction-of-arrival (DOA) estimation with an L-shaped array. The pipeline estimates source count, synchronizes source observations, infers a session calibration state from context scenes, selects a guard-conditioned array manifold, and estimates target DOAs with calibrated ALS.

This repository includes MATLAB dataset generators, Python training and inference code, frozen checkpoints, baseline implementations, and scripts for the paper experiments. Large generated MAT files and evaluation outputs are not part of the source release.

## Quick Start

Use Python 3.10 or 3.11. Install a PyTorch 2.5.1 build compatible with your CPU/CUDA environment, then install the remaining dependencies and the local package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

The frozen source-count checkpoint uses scikit-learn 1.6.1; keep that pinned version when loading it.

Place the generated MAT files in `data/`, using the filenames in `configs/final_protocol.json`. To generate them with MATLAB, follow [simulation/README.md](simulation/README.md). If the files already exist outside the repository, map their dataset keys to absolute paths in a local `configs/data_paths.json`; [configs/data_paths.example.json](configs/data_paths.example.json) is the template. The local path file is Git-ignored.

To check all required datasets and checkpoints, then run a small inference example:

```bash
python scripts/validate_environment.py
python scripts/run_inference.py --states 0,1,2
```

The inference example writes predictions to `outputs/inference/`. For the complete paper evaluation, run:

```bash
python scripts/run_paper_experiments.py
```

This command runs the experiment suite, assembles shared result tables, exports figures, and writes `outputs/reproduction/verification_report.json`. It can take substantial time. To inspect or run individual experiments:

```bash
python scripts/run_paper_experiments.py --list
python scripts/run_paper_experiments.py --experiments overall oracle ood --skip-export --skip-verification
```

Use `--skip-export --skip-verification` for partial runs unless the other required experiment outputs already exist. Long evaluations save resumable progress under `outputs/`.

## Code Map

| Path | Purpose |
| --- | --- |
| `src/sc_phdoa/source_count.py` | Source-count feature extraction and estimator interface. |
| `src/sc_phdoa/synchronization.py` | Source synchronization and mixing estimates. |
| `src/sc_phdoa/features.py`, `encoders.py`, `calibration.py` | Analytic/IQ features and session-calibration network. |
| `src/sc_phdoa/guard_features.py`, `guard_routing.py` | Guard features, residual correction, and route selection. |
| `src/sc_phdoa/manifold.py`, `als.py` | Array manifold construction and calibrated ALS. |
| `src/sc_phdoa/inference/` | Session-level inference pipeline and prediction outputs. |
| `src/sc_phdoa/baselines/` | Classical, EM-SBL, Cov-SetNet, and IQ-SetNet comparison methods. |
| `src/sc_phdoa/metrics.py`, `evaluation.py` | Evaluation metrics and result summaries. |
| `simulation/` | MATLAB generators for training, external ID, OOD, close-source, and calibration-mismatch datasets. |
| `scripts/` | Training, inference, validation, experiment orchestration, and result export. |
| `experiments/` | Individual paper experiment entry points; see [experiments/README.md](experiments/README.md). |
| `configs/final_protocol.json` | Dataset names and evaluation settings. |
| `checkpoints/` | Frozen model and baseline artifacts used by the evaluation scripts. |
| `reference_results/` | Expected metrics for post-evaluation verification; not inference inputs. |
| `tests/` | Architecture, protocol, and release checks. |

## Training Entry Points

The bundled checkpoints are sufficient for evaluation. To retrain the learned components from generated training MAT files, the principal scripts are:

| Script | Output |
| --- | --- |
| `scripts/train_source_count.py` | Source-count classifier and selection records. |
| `scripts/train_session_calibration.py` | Session-calibration network and reusable training cache. |
| `scripts/train_guard_residual.py` | Guard residual model and training features. |
| `scripts/select_guard_threshold.py` | Validation-selected routing threshold. |

These scripts write to `outputs/` and do not overwrite the bundled checkpoints automatically. Training for experiment-specific models is described in [experiments/README.md](experiments/README.md).

## Results

Each experiment writes its own files under `outputs/<experiment>/`. Shared comparison and breakdown tables are assembled in `outputs/assembled_paper_results/`; publication figures and tables are exported to `outputs/paper_artifacts/`. `scripts/verify_reproduction.py` checks output completeness and numerical agreement with `reference_results/` after the full run.
