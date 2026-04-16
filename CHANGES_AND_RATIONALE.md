# Polycube Solver: Changes And Rationale

This document summarizes the major project changes made during the recent iteration.
For each change: what existed before, what was changed, and why it was changed.

## 1) `phase1/__init__.py`: Remove heavy import side effects

### Before
- Importing `phase1` also imported visualization and test modules.
- That transitively imported `matplotlib` even when only solver logic was needed.

### Changed
- `phase1/__init__.py` now exports core solver utilities only (`polycube`, `DLX`, `solve`, etc.).
- Visualization/test imports were removed from package init.

### Why
- Prevent environment and startup issues (especially notebook kernel import failures tied to plotting stack).
- Keep package import lightweight

---

## 2) `phase2/nn_solver.py`: Replace random placement truncation with policy-guided deterministic selection

### Before
- When a state had too many valid placements, candidates were reduced via `random.sample(...)`.
- Search expansion was partly random, not model guided.

### Changed
- Added policy-guided top-K placement selection:
  - Score each valid placement using policy logits.
  - Keep highest scoring placements with deterministic tie breaks.
- Removed random truncation behavior.

### Why
- Make NN search behavior reproducible.
- Use learned policy signal to spend beam budget on more promising actions.
- Reduce search noise and evaluation variance.

---

## 3) `phase2/nn_solver.py`: Batch policy inference across the beam

### Before
- Policy scoring during placement pruning was done per state, causing repeated forward passes.

### Changed
- Added batched policy scoring per depth (`_score_beam_policies`): one model pass for all beam states.
- Placement ranking now reuses batched logits

### Why
- improve inference efficiency and scalability with beam width
- Rduce repeated model invocation overhead

---

## 4) `phase2/train.py`: Expose and wire training loss weights at pipeline level

### Before
- Loss weights existed in low-level `train(...)`, but high level training/ADI entry points did not fully expose them for control

### Changed
- `run_supervised_training(...)` and `run_adi_iteration(...)` now accept and pass `lambda_value` and `lambda_policy` explicitly
- Metadata now records these weights

### Why
- Ensure controlled ablations and reproducibility
- Make objective weighting explicit in experiments

---

## 5) `phase2/train.py`: Reduce ADI label noise from failed beam searches

### Before
- Failed beam search traces were labeled as negative (`unsolvable`) by default
- This conflated "beam failed" with "state is unsolvable" and made false negatives

### Changed
- Added configurable failed-state handling in ADI:
  - `failed_label_mode='verify'` (default): sample failed states and verify unsolvability with DLX before labeling negative.
  - `failed_label_mode='skip'`: do not add failed states.
  - `failed_label_mode='negative'`: legacy behavior.
- Added counters and reporting for verified/labeled/skipped failed states.

### Why
- improve supervision quality for value learning.
- reduce false-negative pressure and instability in self-improvement loops.

---

## 6) `hybrid_solver.py`: Enforce real DLX timeout guard

### Before
- `timeout_dlx` was present in the API but DLX fallback effectively ran inline without hard timeout enforcement.

### Changed
- Added timeout-enforced DLX execution using a worker process.
- Hybrid fallback now handles:
  - normal solve/no-solution,
  - timeout,
  - worker error.

### Why
- Throughput safety: no single hard case can stall the run indefinitely.
- Align runtime behavior with configured timeout expectations.

---

## 7) `phase2/data_generator.py` + `phase2/train.py`: Fix evaluation leakage via grouped split

### Before
- Train/val split was row-random at state level.
- States from the same underlying puzzle instance could leak across train and val.

### Changed
- Added `instance_id` tagging to generated examples.
- `split_dataset(...)` now supports grouped splitting (`group_key='instance_id'`) and uses it by default in training pipelines.
- ADI-generated examples also receive `instance_id` for grouped splitting.

### Why
- Make validation metrics reflect generalization, not near-duplicate memorization.
- Improve trustworthiness of reported model quality.

---

## 8) `phase2/data_generator.py` + `phase2/train.py`: Policy target redesign (action-aligned policy supervision)

### Before
- Policy loss target was a proxy derived from voxel `argmax` of `next_placement`.
- Search action = full placement, but supervision target = single cell index (mismatch).

### Changed
- Introduced placement-action candidate supervision:
  - Build valid placement candidates for the next piece in each positive state.
  - Represent each candidate as a normalized action mask over grid cells.
  - Store `policy_candidates` and `policy_target_idx` in examples/dataset.
- Updated training policy loss computation:
  - Convert cell logits to action logits via projection (`candidates @ policy_logits`).
  - Apply CE over valid placement actions.
- Dataset now returns `policy_candidates`, `policy_action_mask`, and `policy_target_idx`.

### Why
- Align policy training objective with actual decision space used in search.
- Remove proxy mismatch that previously made policy training unreliable.

---

## 9) `config.py`: Updated defaults to match empirical robustness

### Before
- `lambda_policy` was toggled during experimentation.

### Changed
- Current default set to `lambda_policy = 0.0`.

### Why
- After action-space redesign, ablation still showed weaker hard-case solve-rate when policy loss weight was high.
- Keeping policy code in place but defaulting to value-only currently yields stronger robustness on harder benchmarks.

---

## 10) `grading_harness.py` (new): Proxy grading framework aligned to assignment goals

### Added
- New standalone harness for grade-oriented testing.
- Features:
  - Deterministic 20-case proxy suite.
  - Oracle labeling via DLX.
  - Solver modes: `hybrid`, `dlx`, `nn`.
  - Per-case correctness validation (including solution validity checks).
  - A+-style threshold flag (`>=15/20`).
  - Solvable-only and unsolvable-only accuracy metrics.
  - Optional exclusion of easy fixed cases (`include_fixed_cases=False`).
  - Multi-seed aggregation (`run_multi_seed_proxy_grading`).

### Why
- Provide a repeatable, grade-relevant readiness signal.
- Detect overfitting/luck from single-seed evaluations.
- Separate hybrid safety-net performance from NN-only capability.

---

## 11) Notebook workflow and reproducibility improvements

### Added/Adopted
- Structured notebook sections for each change.
- Explicit ablation and stress-test cells.
- Seeded evaluation patterns and grouped split checks.

### Why
- Improve collaboration, traceability, and report-ready experimentation.

---

## Current practical interpretation

- Hybrid solver is strong on current proxy grading checks.
- Core engineering risks (timeouts, leakage, noisy ADI labels, non-deterministic pruning) have been reduced.
- Policy supervision is now correctly aligned to placement actions, but policy-loss weighting still needs cautious tuning for hard-case robustness.
