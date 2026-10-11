# Paper Experiment Entry Points

Every evaluator reads `configs/final_protocol.json`, uses repository-relative
data/checkpoint paths, writes atomic CSV checkpoints, and resumes only caches
whose protocol fingerprint matches the current final configuration.

| ID | Experiment | Entry point | Principal outputs |
| --- | --- | --- | --- |
| E01/E06/E07/E09 | Overall and shared breakdowns | `overall_comparison/evaluate.py`, `classical_baselines/evaluate.py`, `em_sbl/evaluate.py`, `scripts/assemble_paper_results.py` | complete baseline table, SNR, K, and impairment summaries |
| E02 | Observation budget | `observation_budget/train.py`, `evaluate.py` | independent 8/12-support SetNets and paired CI |
| E03 | Calibration/manifold | `calibration_manifold/evaluate.py` | component and manifold errors per session |
| E04 | Oracle manifold | `oracle_calibration/evaluate.py` | four-manifold comparison, activation, paired CI |
| E05 | Context availability | `context_sensitivity/evaluate.py` | nested context subsets and manifold/DOA metrics |
| E07 | Oracle source count | `oracle_source_count/evaluate.py` | predicted-K versus target-only oracle-K |
| E08 | Close-source resolution | `close_source_resolution/evaluate.py` | spatial-only and joint-hard resolution summaries |
| E10 | OOD generalization | `ood_generalization/evaluate.py` | ID/mild/strong summaries, manifold reduction, paired CI |
| E11 | Calibration mismatch | `calibration_drift/evaluate.py` | four routes, four components, five drift strengths |
| E12 | Shuffled session | `shuffled_session/evaluate.py` | within-condition derangement and paired CI |
| E13 | Calibration-input ablation | `calibration_input_ablation/train.py`, `evaluate.py` | analytic-only, IQ-only, and full independently trained networks |
| E14 | Runtime | `runtime_benchmark/benchmark.py` | stratified wall-clock/module latency, parameters, memory |
| E15 | ALS depth | `als_depth_validation/evaluate.py` | 0/1/3/5 steps on the original validation split |

Frozen ablation and controlled-baseline checkpoints are bundled. Retraining is
optional; all frozen-checkpoint evaluations are launched by:

```text
python scripts/run_paper_experiments.py
```

Intermediate prediction arrays are not written during bulk evaluation. The
compact per-scene CSV is sufficient for every reported aggregate and avoids
large, redundant output directories.
