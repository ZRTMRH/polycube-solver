# Next Steps: 12x12x12 Scaling + Branch Merge Plan

**Context:** Professor confirmed the grading test can use grids up to 12x12x12
with pieces of size 3, 4, or 5. This document analyzes what that means for the
project direction and lays out a merge/integration plan for the team's branches.

Written 2026-04-15.

---

## Part 1 — Scale reality check

### The size jump is qualitative, not just quantitative

| Grid | Cells | Typical pieces (size 3–5) | Relative to 3x3x3 |
|------|------:|--------------------------:|------------------:|
| 3x3x3  |   27 |       5–9  |    1× |
| 4x4x4  |   64 |      13–22 |  ~2.4× |
| 5x5x5  |  125 |      25–42 |  ~4.6× |
| 12x12x12 | 1728 | **346–576** | **64×** |

Search depth, beam memory, training sample size, and NN input tensor size all
scale with piece count. A strategy tuned for 3–5 cube grids will not transfer.

### Verified technical blockers for 12x12x12

I read the actual code on `origin/codex/5x5-code-clean` to confirm these. They
are not speculation:

**1. `CuboidNet` is not scale-invariant.**
`phase2/nn_model.py` uses `nn.Linear(32 * grid_size**3, 256)` in the value head
and `nn.Linear(64 * grid_size**3, 512) -> Linear(512, grid_size**3)` in the
policy head. The conv trunk is scale-invariant, but the FC heads have weight
matrices whose shapes are bound to `grid_size`. **A model trained at 4x4x4
cannot be loaded at 12x12x12.** Training a fresh model at 12x12x12 means a
policy-head output layer of 1728 units and FC input layers of ~3M features.

**2. State encoding is memory-hostile at this scale.**
`encode_remaining_pieces` in `phase2/data_generator.py` builds a tensor of
shape `(max_pieces, N, N, N)`. At 12x12x12 with `max_pieces=600` that is
~1.04M floats per state. Each piece channel encodes a 3–5 cell shape in a
1728-cell volume — 99.7% of values are zero. This is wasteful and will dominate
beam-search memory.

**3. DLX fallback stops being a safety net.**
DLX matrix rows ≈ (avg placements per piece) × (num pieces). In a 12x12x12
grid, a size-5 piece has on the order of 10^4 placements across 24 rotations
and N^3 translations. With ~500 pieces you're looking at matrices of millions
of rows. The `_run_dlx_with_timeout` guard just means you hit the timeout,
not that you get a solution. The hybrid solver's core assumption — NN is
approximate, DLX is exact oracle — does not hold at this scale.

**4. The current grading harness doesn't test this regime.**
`grading_harness.build_proxy_suite` only generates 2x2x2 and 3x3x3 cases.
Nick's `build_scale_suite` extends to higher grid sizes via the constructive
generator, but nothing has actually been benchmarked at 12x12x12 yet.

### But 12x12x12 is also *structurally easier* than the small cases

With 1728 cells and pieces of just 3–5 cells, the puzzle has enormous slack.
Most valid piece sets are solvable by almost-greedy placement. The hard parts:
(a) specific adversarial piece combinations the grader may construct,
(b) endgame — filling the last 10–30 cells with whatever pieces remain.

**Implication:** strong classical heuristics may beat an expensive NN at this
scale. The DeepCube-style value-and-policy network is motivated by Rubik's-cube-like
combinatorial texture where each move matters; placing a 3-cell polycube in
a mostly-empty 12^3 grid does not have that texture until very late in the game.

---

## Part 2 — Recommended strategic pivot

**Do not train a dedicated 12x12x12 neural net.** It's expensive, won't
generalize, and likely won't beat a well-engineered classical solver at this
scale.

**Do split the work into three tiers:**

| Tier | Grids | Primary approach |
|------|-------|------------------|
| Small | 2–4 | Existing hybrid (NN-first + DLX fallback). Already works. |
| Medium | 5–7 | NN-guided beam search with connectivity pruning. Current 5x5 work. |
| **Large** | **8–12** | **Block decomposition + greedy + backtracking.** New focus. |

### The large-tier strategy: block decomposition

12³ = 8 × 6³ = 27 × 4³ = 64 × 3³. You can tile the 12x12x12 grid with
disjoint sub-cubes and solve each sub-cube independently.

Pseudo-algorithm:

```
1. Choose block size B (B=6 gives 8 blocks, each 216 cells).
2. Bin-pack the piece list into 8 groups whose volumes sum to 216 each.
   (This is itself a small 1D bin-packing problem — tractable.)
3. For each block, solve it as an independent sub-cube-packing problem
   using the existing medium-tier solver.
4. If any block is infeasible, repartition pieces and retry.
```

Nick's `hybrid_solver.py` already has a `block_planner`. Elevate it from
fallback to primary for the large tier. The hard part is piece-to-block
assignment; the block-solving itself is just the existing medium-tier problem.

Alternative: recursive decomposition (solve top-half 12×12×6, then bottom-half).
More flexible but harder to partition pieces.

### Immediate baseline to establish

Before any more model training, write and benchmark a **pure classical solver**
at 12x12x12:

- MRV piece ordering (already implemented).
- Connectivity pruning (Isabelle's contribution, already merged on 5x5 branch).
- Random restarts with time budget.
- No NN, no DLX.

If this hits ~80%+ on Nick's constructive suite at 12x12x12, you have your
baseline for the large tier, and the NN becomes optional polish.
I suspect it will be surprisingly strong. Measure before optimizing.

### If you want to keep an NN in the large tier

Make it scale-invariant. That means replacing the FC heads in `CuboidNet`:

- **Value head:** global-average-pool the conv trunk, then a small FC head
  operating on pooled features (size independent of N). This is a standard
  modification and preserves the learned filters.
- **Policy head:** output a per-voxel logit via a final 1×1×1 conv
  (shape `(batch, 1, N, N, N)`), then flatten. This is already spatial and
  scale-invariant if you drop the intermediate FC.

With these changes, a model trained on 5x5x5 and 6x6x6 data can be *applied*
at 12x12x12 without retraining. You only use it as a placement-ranking
heuristic inside each block; not for solving the full 1728-cell puzzle at once.

The remaining-pieces channel encoding is still wasteful at large N. A cheaper
alternative: instead of one 3D channel per remaining piece, use a small set of
global features (count by size, count by piece type after canonicalization,
etc.) concatenated via FiLM or broadcast channels. That removes the
`max_pieces × N³` blow-up.

### What to keep from current work

Almost all of it is reusable:

- **Isabelle's connectivity pruning** — essential at 12x12x12. Keep.
- **Nick's constructive case generator** — the only sound way to build a
  12x12x12 benchmark. Keep and use immediately.
- **Nick's search profiles per grid size** — already designed for tiered
  strategies. Extend with a `large` tier profile.
- **Nick's block planner** — promote to primary for large tier.
- **Nick's diagnostics** (`analyze_5x5_bottlenecks.py`, etc.) — directly
  portable to 12x12x12 once there's something to diagnose.
- **Rally's Modal GPU setup** — useful for training the scale-invariant
  model if you go that route. Not urgent for the large tier itself.

### What to stop investing in

- **Grid-size-specific model checkpoints.** Every `.pt` file tied to a
  specific N is a sunk cost that won't help at 12x12x12.
- **Tuning the 5x5 search profile further.** Diminishing returns; time is
  better spent on the large tier.
- **Treating hybrid = NN-then-DLX as the universal strategy.** Replace DLX
  fallback with block decomposition for large grids.

---

## Part 3 — Branch merge plan

### Current state

```
master (your initial commit, f5655ed)
 └─ b059d53  codex/full-project-updates         [Codex partner]
    └─ 17a9ec5→382cb01→fe74af9→60a4ca9          [Isabelle: connectivity+4x4]
       ├─ isd8-updates-1                         (Isabelle's tip, superseded)
       ├─ 74d7ea5  rally/modal-gpu               [Rally: 60-line T4 inference]
       └─ f88cc3f  codex/5x5-code-clean          [Nick: 5x5 + big refactor]
```

Active tips: `rally/modal-gpu` and `codex/5x5-code-clean`. Neither subsumes
the other:

- `codex/5x5-code-clean` does NOT contain Rally's `modal_solve_test.py`.
  Verified: `git diff --stat origin/rally/modal-gpu origin/codex/5x5-code-clean`
  shows `modal_solve_test.py | 60 --`.
- `codex/5x5-code-clean` has a more extensive `modal_train.py` (Nick's version
  adds Modal Volume integration — +183 lines vs Rally's 50). These will
  conflict on merge. Nick's version is the newer and more capable one.

### Recommended merge plan

**Goal:** a single integration branch that has everything, which you can use
as the starting point for the 12x12x12 work.

**Step 1 — Create integration branch from Nick's work.**
```bash
git switch -c integration origin/codex/5x5-code-clean
```
This gets you everyone's work except Rally's `modal_solve_test.py`.

**Step 2 — Cherry-pick Rally's commit.**
```bash
git cherry-pick 74d7ea5
```
Expect one conflict in `modal_train.py`. **Resolve by keeping Nick's version**
(it's a superset) — discard Rally's changes to that file. The new file
`modal_solve_test.py` will apply cleanly.

```bash
# during conflict resolution:
git checkout --ours modal_train.py      # keep Nick's version
git add modal_train.py modal_solve_test.py
git cherry-pick --continue
```

**Step 3 — Verify imports still work.**
```bash
~/venv/bin/python3 -c "import phase2.nn_solver, phase2.train, hybrid_solver, grading_harness"
```
If this passes, the merge is structurally clean.

**Step 4 — Run the existing 5x5 benchmark as a regression check.**
Before touching anything, make sure the existing behavior is preserved.
If Nick has a standard benchmark command (see his diagnostic scripts), run it.

**Step 5 — Push integration branch.**
```bash
git push -u origin integration
```

**Step 6 — Do NOT fast-forward master yet.** Master is intentionally empty.
Promote `integration` to master only once the 12x12x12 work is at least
baselined — otherwise master becomes a snapshot of work-in-progress with
no clear reason to be there.

### On the binary model files

The repo is carrying 6 `.pt` files totaling ~180MB committed to git history.
These should move out before the repo grows further:

- Best option: Modal Volume (already set up in Nick's `modal_train.py`).
- Acceptable: Git LFS.
- Bad: leaving them in `.git` history.

This is cleanup, not urgent — note it, don't block on it.

---

## Part 4 — Concrete next-week plan

In priority order:

1. **Build the 12x12x12 benchmark.** Use Nick's `build_constructive_case`
   to generate 20 solvable 12x12x12 cases. Save them to a fixture file so
   results are reproducible across runs.

2. **Measure the pure-classical baseline on 12x12x12.** MRV + connectivity
   pruning + random restarts + time budget. No NN. Report solve rate and
   avg time. This is the reference point everything else must beat.

3. **Prototype block decomposition.** Wrap the existing medium-tier solver
   in an 8-block driver. Compare against the classical baseline from (2).

4. **Decide whether the NN is worth keeping in the large tier.** Only
   invest in the scale-invariant architecture refactor if (2) and (3)
   leave a clear gap the NN could close.

5. **Merge branches into `integration`** per Part 3. Do this early so
   subsequent work builds on a single tip.

Everything else — policy-loss tuning, adversarial test generation, ADI
iterations — is polish. The 12x12x12 benchmark and block decomposition
prototype are the highest-leverage work this week.

---

## Appendix — Commands reference

Inspect branches:
```bash
git log --graph --oneline --all
git diff --stat origin/rally/modal-gpu origin/codex/5x5-code-clean
```

Switch to integration starting point:
```bash
git fetch --all --prune
git switch -c integration origin/codex/5x5-code-clean
git cherry-pick 74d7ea5   # expect conflict in modal_train.py
```

Key files to know:
- `phase2/nn_model.py` — CuboidNet (FC heads are scale-bound)
- `phase2/data_generator.py:744` — `encode_remaining_pieces` (memory hot spot)
- `phase2/nn_solver.py` — beam search, `has_isolated_pockets` (connectivity)
- `hybrid_solver.py` — block planner lives here
- `grading_harness.py` — `build_constructive_case`, `build_scale_suite`
- `phase2/search_profiles.py` — tiered search configuration
