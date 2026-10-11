# Simulation Data Generation

The seven MATLAB sources in this folder are the released generators and their
shared waveform helper. `data/` is output-only; it contains no MATLAB source.
Run MATLAB from the repository root in this order:

```matlab
addpath('simulation')
run('simulation/generate_source_count_training.m')
run('simulation/generate_session_calibration_training.m')
run('simulation/generate_external_session_test.m')
generate_ood_tests
generate_close_source_tests
generate_calibration_drift_tests
```

The first two commands create the training MAT files. The external generator
reads only the fixed shared preamble from `session_calibration_training.mat`;
the remaining test generators use that same preamble. The complete sequence
creates 11 MAT files: two training sets, external ID, two OOD, two close-source,
and four calibration-mismatch sets. Generation can take substantial time and
disk space; no model training occurs in MATLAB.

All programs use fixed seeds and write directly to `data/` using the names in
`configs/final_protocol.json`. Existing final or `.partial.mat` files are not
overwritten. This prevents an interrupted or accidental second run from being
mistaken for the locked data realization.

After generation, run `python scripts/validate_environment.py` to check the
MAT schemas and frozen checkpoints before evaluating the paper experiments.
For a code-only smoke check without generating large MAT files, add
`simulation/` to the MATLAB path and call `generate_structured_waveform`.

The external test generators reuse only the fixed shared preamble stored in
`session_calibration_training.mat`; they do not read training IQ samples or
training labels.
