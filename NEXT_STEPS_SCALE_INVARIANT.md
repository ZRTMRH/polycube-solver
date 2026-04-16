# Scale-Invariant CuboidNet Training Plan

## Audit summary (what's already in place)

- **Model**: `phase2/nn_model.py::CuboidNet` already supports `value_head_type='gap'`,
  `policy_head_type='conv'`, `use_context_features=True`. With these flags,
  zero parameters depend on `grid_size`. Verified by `test_scale_invariant_forward.py`:
  one model evaluates cleanly at N=5/7/9/12, save/load round-trip preserves
  outputs.
- **Recipes**: `phase2/training_recipes.py::recommended_training_recipe`
  already returns `gap`/`conv`/context for `grid_size >= 5` (and even for
  grid_size=4). Defaults look fine: 160 hidden / 8 blocks / 45 epochs / 160
  instances at N>=5.
- **Modal entrypoint**: `modal_train.py` already plumbs `value_head_type`,
  `policy_head_type`, `use_context_features` overrides through to
  `run_supervised_training`, and the saved checkpoint blob includes those
  fields. `phase2/train.py::load_model` reads them back. Plumbing is end-to-end
  correct.
- **Data generator**: `generate_constructive_puzzle_instances` natively
  handles grid_size 5, 7, 9, 12 via `_build_mixed_constructive_solution` /
  `_build_striped_constructive_solution`.

## What needs changing

Training already works at any single grid size with scale-invariant heads.
The one real gap: `phase2/data_generator.py::create_torch_dataset` stacks all
example states into a single tensor, so a *single* `DataLoader` cannot mix
N=5 and N=12 examples (shapes won't broadcast).

### Recommended minimal path: sequential curriculum

Skip multi-grid mixing. Train **sequentially** on one grid at a time, each
fine-tuning the previous checkpoint. Concretely:

1. Train fresh at **N=5** (cheap, fast). Save as `scaleinv_v1_n5.pt`.
2. Resume from N=5 weights, fine-tune at **N=7**. Save as `scaleinv_v1_n7.pt`.
3. Resume from N=7 weights, fine-tune at **N=9**. Save as `scaleinv_v1_n9.pt`.
4. Resume from N=9 weights, fine-tune at **N=12** (small num_instances since
   constructive generation is slower at large N). Save as `scaleinv_v1_n12.pt`.

Curriculum from small to large is a standard trick and avoids the dataset
shape problem entirely. Because heads are scale-invariant, weights transfer.

The only patch required is **adding a `--resume_from` option to
`modal_train.py::train`** that loads a prior checkpoint's `model_state_dict`
into the freshly-built model before calling `train()`. This is ~10 lines:

```python
# inside train(), after `run_supervised_training` builds the model — actually
# better to refactor: build model first, optionally load weights, then pass
# into run_supervised_training. Or add an `init_state_dict_path` kwarg.
```

If we don't want to touch the training pipeline, an even simpler hack: train
each stage independently from scratch with only the largest stage being long
— scale-invariant heads transfer poorly without resume, but a pure N=12 run
(with a few hundred small constructive instances) will already give us a
heuristic that beats nothing.

### Even-more-minimal "ship it" path (recommended first attempt)

Just train **once at N=9** with the existing recipe. N=9 sees enough capacity
strain to learn meaningful global features, and the gap/conv heads will
generalize down to N=5/7 and up to N=12 without retraining. Total cost
roughly equal to one current Modal run.

## Modal CLI command (single N=9 run)

```bash
modal run modal_train.py \
    --grid-size 9 \
    --max-pieces 30 \
    --hidden-dim 160 \
    --num-residual-blocks 8 \
    --use-context-features true \
    --epochs 45 \
    --num-instances 160 \
    --save-name scaleinv_v1_n9
```

(Recipe defaults already give `value_head_type='gap'` and
`policy_head_type='conv'` for `grid_size >= 5`, so no need to override those.)

For the curriculum variant, add `--resume_from <prior_checkpoint>` after
patching `modal_train.py`, then run for N=5, N=7, N=9, N=12 in order with
shorter `--epochs` (e.g. 20) for the fine-tuning stages.

## Compute / cost estimate (T4, order of magnitude)

Existing 4x4x4 run (~80 instances, 35 epochs, 6 blocks @ 128-dim) takes a
T4 about 15-25 minutes. Scaling to N=9 with 160 instances, 45 epochs, 8
blocks @ 160-dim:

- Forward FLOPs scale as N^3 (~11x more voxels than N=4) and channels^2
  (~1.6x). Per-batch ~17x cost, but batch size halves to keep memory sane.
- Total: roughly **2-4 hours** on a T4. **Cost: ~$1-3** at Modal T4 pricing
  ($0.59/hr).

For the curriculum: 4 stages, dominated by the N=12 stage (~3-5 hours since
constructive generation is also slow). Total **~5-8 hours, ~$3-5**.

## Risk assessment

- **Will the model train successfully?** High confidence (>90%). All the
  plumbing exists; the dry-run test confirms forward passes at every target
  N work.
- **Will it produce a useful value estimate at N=12?** Medium confidence
  (~50%). Constructive instances at N=12 are easy (always solvable by
  construction), so the value head will learn "fill ratio + remaining
  volume" approximately. Without negatives at large N, the value head is
  near-useless as a discriminator; consider `--num-negatives 1` at the
  largest stages.
- **Will it improve solver accuracy at 12³?** Low-to-medium confidence
  (~25-40%). The NN is a heuristic inside beam search — if beam search at
  12³ is bottlenecked by **branching factor / piece-placement enumeration**
  rather than node ordering, a better value estimate buys very little. The
  audit notes already flag that beam search itself may be the bottleneck;
  spending $5 to find that out is reasonable, but don't expect miracles.

**Recommended next action**: ship the single-N=9 run first (cheap test of
the whole pipeline). If the resulting checkpoint helps the solver at N=7-9
in fixture benchmarks, then invest in the full curriculum to push N=12. If
it doesn't help at N=7-9, the bottleneck is search, not the heuristic, and
the curriculum will not save us.
