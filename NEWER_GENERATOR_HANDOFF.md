# Newer Generator Handoff

This repo now contains the checkpoints and scripts for the newer constructive
generator regime. The old DLX-style hard family is not the target here.

## Best checkpoints

- 5x5 newer-generator model:
  `phase2/trained_models/5x5x5_calibrated_v1_light.pt`
- 6x6 newer-generator model:
  `phase2/trained_models/6x6x6_frontier_adi_v1.pt`

## Newer-generator benchmark results

- 5x5 greedy_random:
  `reports/5x5_v1_light_greedy_t24_20.json`
  - 19/20 = 95%
- 5x5 robust_rotated:
  `reports/5x5_v1_light_robust_rotated_t24_10.json`
  - 9/10 = 90%
- 6x6 constructive-family benchmark:
  `reports/6x6x6_frontier_adi_v1_eval.json`
  - main_connected: 18/20 = 90%
  - new_hard: 9/10 = 90%

The 6x6 eval file also includes `legacy_style_hard`, which is the older harder
family. That bucket is not the intended headline metric for this handoff.

## Benchmark suites

- 6x6 representative constructive suite:
  `reports/6x6_benchmark_suite_v2.json`
- generic constructive-only scale suite builder:
  `build_constructive_scale_suite.py`
- Modal wrapper for constructive-only suites:
  `modal_constructive_scale_suite.py`

## Common commands

### 5x5 robust newer-generator benchmark

```bash
.venv/bin/python -u benchmark_robust_generated_cases.py \
  --model-name 5x5x5_calibrated_v1_light \
  --grid-size 5 \
  --eval-cases 10 \
  --timeout-nn 24 \
  --timeout-dlx 0.1 \
  --device cpu \
  --out reports/5x5_v1_light_robust_rotated_t24_10.json
```

### 6x6 fixed-suite benchmark

```bash
.venv/bin/modal run modal_benchmark_fixed_suite.py \
  --model-name 6x6x6_frontier_adi_v1 \
  --suite-path reports/6x6_benchmark_suite_v2.json \
  --timeout-nn 24 \
  --timeout-dlx 0.1 \
  --save-name 6x6x6_frontier_adi_v1_eval
```

### Warm-start a larger model

```bash
.venv/bin/python warmstart_model.py \
  --source-name 6x6x6_frontier_adi_v1 \
  --target-name 7x7x7_warmstart_from_6x6_constructive \
  --grid-size 7 \
  --max-pieces 96
```

## Important code paths

- checkpoint compatibility / transfer loading:
  `phase2/train.py`
- resized inference loading:
  `hybrid_solver.py`
  `phase2/nn_solver.py`
- constructive generation:
  `phase2/data_generator.py`
- 6x6/7x7/8x8 search settings:
  `phase2/search_profiles.py`
- Modal frontier ADI:
  `modal_frontier_adi.py`
