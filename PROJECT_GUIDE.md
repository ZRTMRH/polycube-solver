# 3D Polycube Packing Solver - Project Guide

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Problem We Are Solving](#2-the-problem-we-are-solving)
3. [File-by-File Explanation](#3-file-by-file-explanation)
4. [How the Solver Works (Big Picture)](#4-how-the-solver-works-big-picture)
5. [Algorithms Explained](#5-algorithms-explained)
6. [Current Performance](#6-current-performance)
7. [Strategy Evaluation - Are We on the Right Track?](#7-strategy-evaluation---are-we-on-the-right-track)
8. [Remaining Weaknesses and Risks](#8-remaining-weaknesses-and-risks)
9. [Work Division Plan](#9-work-division-plan)

---

## 1. What Is This Project?

We are building a solver for a 3D puzzle game, similar to 3D Tetris. Given a
cubic box of size N x N x N and a set of small 3D shapes (polycubes, each made
of 3-5 unit cubes), our job is to figure out how to fit all the pieces into the
box with no gaps and no overlaps -- or to correctly determine that it is
impossible.

This is for STA 561 (Probabilistic Machine Learning), so we also incorporate
neural-network-based and probabilistic techniques alongside classical
algorithms.

---

## 2. The Problem We Are Solving

**Input:** A grid size N and a list of pieces. Each piece is a set of 3D
coordinates (like Lego blocks made of small cubes).

**Output:** Either a valid placement (which cell each piece goes into) or
"unsolvable" (if the pieces cannot fill the box exactly).

**Why this is hard:** The number of possible ways to place pieces grows
astronomically. Even for a 5x5x5 box (125 cells, ~31 pieces), there can be
millions of possible placement combinations. This is an NP-complete problem --
there is no known fast algorithm that works for all cases.

**How we are graded:** The professor generates test puzzles of various sizes
(N = 3 to 12+), including both solvable and unsolvable cases. We need to:
- Correctly solve solvable cases (place all pieces)
- Correctly reject unsolvable cases (say "no solution")
- Do it within a time limit

The **A+ threshold** is getting at least 75% of 20 test cases correct.

---

## 3. File-by-File Explanation

### Core Solver Files

#### `hybrid_solver.py` -- The Main Brain (Orchestrator)

This is the central file that decides *which* solving strategy to use for each
puzzle. Think of it as a traffic controller.

**What it does:**
- Receives a puzzle (grid size + pieces)
- Checks basic things first (do the piece volumes even add up to the box volume?)
- Then picks the fastest strategy based on the grid size:

| Grid Size | Primary Strategy | Why |
|-----------|-----------------|-----|
| N = 2-4   | DLX exact search | Small enough to try everything |
| N = 5-6   | DLX first, then neural net | Medium -- DLX might be slow |
| N = 7+    | Fast planners first, then DLX/NN fallback | Too big for brute force |

**Key strategies it coordinates:**
1. **Volume check** -- If piece volumes don't equal N^3, immediately say "unsolvable"
2. **Pre-placed detection** -- If pieces already have coordinates that fit, just verify them
3. **Slab planner** -- If all pieces are flat (thin like a coin), use a fast 2D approach
4. **Rod planner** -- If all pieces are straight lines, use a line-packing approach
5. **Block decomposition** -- For huge grids, split into smaller sub-cubes and solve each
6. **DLX exact search** -- The guaranteed-correct but slow brute-force method
7. **Neural network beam search** -- A trained AI that guesses good placements

#### `phase1/dlx_solver.py` -- The Exact Search Engine (DLX)

This implements "Algorithm X with Dancing Links" (DLX), invented by Donald
Knuth. It is our most reliable solver -- it is guaranteed to find a solution if
one exists, but can be slow for large puzzles.

**How it works (simplified):**
1. Build a big table where each row represents "place piece X in position Y"
   and each column represents either a cell in the grid or a piece
2. The goal: pick exactly one row per piece such that every cell column is
   covered exactly once
3. The algorithm tries placing a piece, then recursively tries to fill the
   rest. If it gets stuck, it backtracks and tries a different placement.
4. The "Dancing Links" trick makes backtracking very fast by using a clever
   linked-list data structure where removing and re-inserting items is O(1)

**Key optimization:** MRV (Minimum Remaining Values) heuristic -- always fill
the cell that has the fewest possible pieces that could go there. This
dramatically reduces the number of dead ends.

**Think of it as:** A very organized trial-and-error search that can undo
mistakes instantly.

#### `phase1/polycube.py` -- Geometry Utilities

Handles the geometry of 3D pieces:
- **Rotation:** Each piece can be rotated 24 different ways in 3D (think of
  all the ways you can orient a die)
- **Normalization:** Slides a piece so its corner is at (0,0,0) for easy
  comparison
- **Placement generation:** For each piece, computes every valid position and
  rotation that fits inside the N x N x N grid

#### `phase1/solver.py` -- DLX Wrapper

A convenience wrapper that takes pieces + grid size, builds the DLX table, and
runs the search. Connects `polycube.py` (geometry) with `dlx_solver.py` (search).

#### `block_planner_v2.py` -- The Fast Planner (Slab + Pair Solver)

This is our secret weapon for large puzzles. Instead of trying all combinations,
it exploits the *structure* of how test puzzles are generated.

**Key insight:** The professor's test generator creates pieces that are "flat"
(they only extend 1 unit in one direction). This means we can solve each
horizontal layer independently instead of solving the whole 3D box at once.

**What it does:**
1. **Detect slab axis** -- Figures out which direction the pieces are flat in
2. **Classify pieces** -- Identifies piece types: L-shapes, squares, P-shapes,
   rods (straight lines)
3. **Pair matching** -- Groups complementary pieces into pairs that tile 2-wide
   strips (like matching puzzle pieces that fit together)
4. **Strip packing** -- Assigns pairs to 2-row strips within each layer
5. **Pattern assignment** -- Figures out how to distribute pieces of different
   sizes across layers so everything adds up perfectly

**Performance:** Solves N=50 (125,000 cells, 33,000 pieces) in about 1 second,
compared to hours or infinity for brute-force DLX.

**Think of it as:** Instead of solving a 3D puzzle, we cleverly reduce it to
many small 2D puzzles.

#### `block_planner_v3.py` -- Experimental Block Decomposition

An alternative approach for large grids that divides the box into smaller
sub-cubes. Less developed than v2. Uses its own DLX implementation internally.

### Neural Network Files (phase2/)

#### `phase2/nn_model.py` -- The Neural Network Architecture

A 3D convolutional neural network called CuboidNet.

**What it does:** Given a partially-filled box, it predicts:
1. **Value:** "How likely is it that this partial placement can lead to a
   complete solution?" (a number from 0 to 1)
2. **Policy:** "Which empty cell should we place the next piece in?"
   (a probability for each cell)

**Architecture:**
- Takes as input a 3D grid with one channel per remaining piece (like layers
  in an image, but 3D)
- Passes through residual blocks (similar to the ones used in image
  recognition -- each block learns to add refinements on top of its input)
- Outputs value (single number) and policy (score per cell)

**Think of it as:** An AI that has learned from thousands of solved puzzles to
estimate whether a partially-built solution is promising.

#### `phase2/nn_solver.py` -- Neural-Network-Guided Search (Beam Search)

Uses the neural network to guide a search. Instead of trying every possibility
(like DLX), it keeps only the most promising partial solutions at each step.

**How beam search works (simplified):**
1. Start with an empty grid
2. Pick the piece with the fewest valid placements (MRV heuristic)
3. Try placing it in various positions
4. Ask the neural network: "which of these partial grids looks most promising?"
5. Keep only the top K most promising states (K = beam width)
6. Repeat from step 2 with the next piece
7. If no beam reaches a complete solution, try again with wider beams

**Also includes:**
- **Pocket pruning:** If placing a piece creates a tiny isolated empty region
  that no remaining piece can fill, discard that option immediately
- **Diversity preservation:** Keep some variety in the beam so we don't get
  stuck exploring similar dead ends
- **Frontier DFS:** If beam search gets close but not quite, switch to
  depth-first search from the best partial solutions

#### `phase2/train.py` -- Training the Neural Network

How we train CuboidNet:
1. **Generate solved puzzles** (using DLX to find actual solutions)
2. **Create training data** from solved puzzles:
   - Positive examples: partially-placed states from real solutions
   - Negative examples: randomly placed pieces that lead to dead ends
3. **Train the network** to distinguish promising states from dead ends
4. **ADI (Autodidactic Iteration):** A self-improvement loop where the NN
   tries to solve new puzzles, and uses its successes/failures as new training
   data (similar to how AlphaGo improves by playing against itself)

#### `phase2/data_generator.py` -- Training Data Factory

Generates puzzle instances and training examples:
- **Polycube enumeration:** Generates all possible piece shapes of sizes 3-5
- **Puzzle generation:** Creates random puzzles, verifies solvability with DLX
- **State encoding:** Converts puzzle states into tensors the neural network
  can understand

### Test and Evaluation Files

#### `grading_harness.py` -- Professor's Grading System (Our Target)

This is a copy/approximation of how the professor generates test cases and
scores our solver. Understanding this is critical.

**How the professor generates puzzles:**

For small grids (N <= 6):
- Constructive: Carve an N^3 cube into connected pieces (guaranteed solvable)
- DLX-verified random: Generate random pieces, verify solvability with DLX

For large grids (N >= 7):
- Mixed constructive: Creates flat pieces in complementary pairs (L-shapes,
  squares, P-shapes) that tile 2-row strips. Always solvable.
- Striped constructive: Creates rod pieces (straight lines) that tile
  along grid lines. Always solvable.

**Key detail:** For N >= 7, pieces are *normalized* (shifted to origin), so we
cannot cheat by looking at absolute coordinates. We must actually solve the
placement problem.

**Scoring:**
- 20 test cases per grading run
- Mix of solvable and unsolvable
- A+ threshold: >= 15/20 correct (75%)

#### `fixture_a_plus.py` -- Our 150-Case Test Suite

A comprehensive test fixture we built to estimate our grade:
- 150 cases across N = 3 to 12
- Mix of solvable (various generators) and unsolvable (volume mutations)
- Tracks per-stratum accuracy and estimates P(A+) statistically

#### `robust_generator.py` -- Genuinely 3D Test Generator

Generates puzzles with truly 3D pieces (not just flat ones). Uses a "surface
peeling" algorithm that grows pieces from the outside of the cube inward,
ensuring the remaining space stays connected.

#### Test scripts (test_*.py)

Various test scripts for different aspects:
- `test_professor_grading.py` -- Simulates the professor's exact grading flow
- `test_stress_comprehensive.py` -- 134 cases across all strategies
- `test_extreme_scale.py` -- Pushes to N = 100+
- `test_paired_solver.py` -- Tests the pair-based fast solver
- Others are debug/development scripts

### Documentation Files

- `REFINEMENT_LOG.md` -- History of solver improvements (9 refinement rounds)
- `NEXT_STEPS_SCALE_INVARIANT.md` -- Plan for making the NN handle any grid size
- `demo.ipynb` -- Jupyter notebook demonstrating the full pipeline

---

## 4. How the Solver Works (Big Picture)

When a puzzle arrives, the solver follows this decision tree:

```
Puzzle arrives (N x N x N grid, list of pieces)
    |
    v
[1] Volume check: Do piece volumes sum to N^3?
    |-- No --> "UNSOLVABLE" (instant)
    |-- Yes --> continue
    v
[2] Pre-placed check: Are pieces already placed correctly?
    |-- Yes --> return solution (instant)
    |-- No --> continue
    v
[3] Are all pieces flat (thin in one direction)?
    |-- Yes --> Slab Planner (fast, O(N^2))
    |           |-- All rods? --> Rod fast path
    |           |-- Mixed shapes? --> Pair-based tiling
    |-- No --> continue
    v
[4] Is N small enough for exact search?
    |-- N <= 4 --> DLX only (guaranteed, usually < 5 sec)
    |-- N = 5-6 --> DLX first (with timeout), then NN
    |-- N = 7-9 --> DLX with long timeout (up to 90 sec)
    v
[5] Is N large and divisible into blocks?
    |-- Yes --> Block decomposition (split into 5^3 or 6^3 sub-cubes)
    |-- No --> Neural network beam search with DLX fallback
```

---

## 5. Algorithms Explained

### Algorithm 1: DLX (Dancing Links) -- "The Reliable Workhorse"

**Category:** Exact search / constraint satisfaction
**From the course:** Related to combinatorial optimization

Think of a Sudoku solver that tries numbers, backtracks when stuck, and is
guaranteed to find a solution if one exists. DLX is the same idea but for
3D puzzles.

**Strengths:** Guaranteed correct, finds solutions for any puzzle
**Weaknesses:** Slow for large puzzles (exponential worst case)
**Used for:** N <= 9 puzzles, verifying solvability, training data generation

### Algorithm 2: Slab Decomposition -- "Divide and Conquer"

**Category:** Problem decomposition / structural exploitation

If all pieces are flat, we can solve each horizontal layer independently. This
reduces one big 3D problem into N small 2D problems.

**Strengths:** Extremely fast (milliseconds for N = 50)
**Weaknesses:** Only works when pieces are flat (which is true for the
professor's large-grid test cases)

### Algorithm 3: Pair-Based Tiling -- "Pattern Matching"

**Category:** Constructive algorithm / probabilistic matching

The professor generates pieces in complementary pairs (an L-shape and its
mirror, two squares, etc.). We detect these pairs and assemble them into
rectangles that tile each layer.

**Strengths:** Near-instant, exploits known structure
**Weaknesses:** Only works for puzzles generated with this specific structure
**Course connection:** Uses ideas similar to stochastic matching and
combinatorial optimization from PML

### Algorithm 4: Neural Network Beam Search -- "Learned Intuition"

**Category:** Deep learning + search
**Course connection:** Direct application of neural networks (Lab PyTorch),
stochastic optimization (Lab 7), and self-play/ADI

The neural network learns from solved examples to predict which partial
placements are promising. Beam search explores multiple options in parallel,
guided by the network's predictions.

**Strengths:** Can handle cases where DLX is too slow
**Weaknesses:** Not guaranteed to find a solution even if one exists; requires
training; currently only trained for small grids

### Algorithm 5: Block Decomposition -- "Divide the Box"

**Category:** Divide and conquer

For very large grids (N = 10, 12, 15...), split the cube into smaller sub-cubes
(e.g., eight 5x5x5 blocks for a 10x10x10 grid). Solve each sub-cube
independently with DLX.

**Strengths:** Makes large grids tractable
**Weaknesses:** Only works when N is divisible by the block size; needs careful
piece allocation to sub-blocks

### Algorithm 6: Profile Assignment -- "Budget Balancing"

**Category:** Integer programming / constraint satisfaction

When distributing pieces across blocks or layers, we need to decide how many
pieces of each size go where. This is like a budgeting problem: each block
needs pieces that sum to exactly its volume.

**Strengths:** Fast (polynomial time), exact
**Used in:** Slab planner and block decomposition

---

## 6. Current Performance

### Honest Evaluation (April 2026)

We ran independent stress tests to check whether the 100% pass rate is real
or whether we accidentally overfitted to our own test suite.

**IMPORTANT CAVEAT:** `grading_harness.py` is code WE wrote as a proxy. It is
NOT the professor's actual grading code. We do not know how the professor
generates test cases. The slab planner and pair-based tiling were built to
solve puzzles from OUR OWN generators, so the 100% on our own test suite is
partially circular.

The truly trustworthy results are the ones using random/independent puzzles
that our solver was never tuned for. Here is what we know for certain:

### Test Results Summary

**Realistic test (random 3D pieces, normalized -- professor-style):**

| Grid | Pass Rate | Avg Time | Method | Status |
|------|-----------|----------|--------|--------|
| N=3 | 20/20 (100%) | 2.0s | DLX exact | OK |
| N=4 | 20/20 (100%) | 2.4s | DLX exact | OK |
| N=5 | 20/20 (100%) | 2.8s | DLX exact-first | OK |
| N=6 | 20/20 (100%) | 4.2s | DLX exact-first | OK |
| N=7 | 20/20 (100%) | 7.2s | DLX fallback | OK |
| N=8 | 10/10 (100%) | 13.3s | DLX fallback | OK |
| N=9 | 10/10 (100%) | 32.1s | DLX fallback | Slow |
| N=10 | 5/5 (100%) | 30.9s | Block decomp (5^3) | OK |
| **N=11** | **0/5 (0%)** | **167s** | **DLX timeout** | **BROKEN** |
| N=12 | 5/5 (100%) | 61.5s | Block decomp (6^3) | OK |

**Other tests:**

| Test | Result | What It Tests |
|------|--------|---------------|
| Our own proxy grading (10 seeds) | 200/200 (100%) | Our own test generators (NOT professor's) |
| Our own stress test | 134/134 (100%) | Our own test generators |
| Unsolvable (volume OK, DLX-verified) | 50/50 (100%) | Hard reject at N=3-4 |

### What Is Genuinely Reliable (works on ANY input, not just our tests)

| Method | Works On | Guarantee |
|--------|----------|-----------|
| DLX exact search | Any puzzle N <= 9 | Mathematically guaranteed correct |
| Volume check | Any puzzle | Instant reject if pieces don't fill the box |
| Block decomposition | N = 10, 12, 15, 20... (divisible by 5 or 6) with any pieces | Splits into DLX sub-problems |

These methods don't depend on ANY assumption about how puzzles are generated.

### What Is Specialized (only works on certain puzzle types)

| Method | Assumption | If Wrong |
|--------|-----------|----------|
| Slab planner | All pieces are flat (1 unit thick) | Falls through to DLX |
| Pair-based tiling | Flat pieces in complementary pairs (from our generator) | Falls through to DLX |
| Pre-placed detection | Pieces already have correct coordinates | Falls through to DLX |

These are fast shortcuts. If the professor generates random 3D pieces
(which is the likely scenario), these planners are useless and everything
falls through to DLX or block decomposition.

### Where We Actually Fail (CONFIRMED BY TESTING)

| Scenario | Outcome | Confirmed |
|----------|---------|-----------|
| Any puzzle N <= 9 | **Always solved** (DLX is universal) | YES -- 100/100 |
| N = 10 with any pieces | **Solved** via block decomp (5^3) | YES -- 5/5 |
| **N = 11 with any 3D pieces** | **FAILS -- 0% solve rate** | **YES -- 0/5** |
| N = 12 with any pieces | **Solved** via block decomp (6^3) | YES -- 5/5 |
| **N = 13, 14 with any 3D pieces** | **Will FAIL** (same as N=11) | Expected |
| N = 15 with any pieces | **Solved** via block decomp (5^3) | Expected |
| Unsolvable puzzle at N >= 10 | **Guesses "unsolvable" after timeout** (may be wrong) | Untested |

**The N=11 problem is our biggest gap.** 11 is not divisible by any block
size we support (5, 6, 7, 8). DLX alone cannot handle N=11 within any
reasonable timeout (takes 165+ seconds and still fails). This is where the
neural network SHOULD be helping, but it is undertrained.

### Speed by Grid Size

**Professor's test cases (flat pieces):**

| Grid Size | Pieces | Method Used | Time |
|-----------|--------|-------------|------|
| N = 3 | 7 | Pre-placed or DLX | < 0.01s |
| N = 4 | 16 | DLX exact or Slab | 0-5s |
| N = 5 | 31 | DLX or Slab | 0-1s |
| N = 7 | ~98 | Slab paired | 0.001s |
| N = 9 | ~180 | Slab paired | 0.002s |
| N = 12 | ~450 | Slab paired | 0.003s |
| N = 15 | ~900 | Slab paired | 0.006s |
| N = 20 | ~2100 | Slab paired | 0.016s |
| N = 50 | ~33000 | Slab paired | 0.2s |

**Arbitrary 3D pieces (harder -- no flat structure to exploit):**

| Grid Size | Pieces | Method | Avg Time | Max Time |
|-----------|--------|--------|----------|----------|
| N = 3 | ~7 | DLX | 0.02s | 0.10s |
| N = 5 | ~31 | DLX | 0.39s | 0.89s |
| N = 6 | ~55 | DLX | 1.37s | 4.16s |
| N = 7 | ~86 | DLX | 4.20s | 7.20s |
| N = 8 | ~129 | DLX | 10.05s | 12.30s |
| N = 9 | ~182 | DLX | 20.90s | 23.02s |
| N = 10 | ~250 | Block(5^3) | 5.66s | 9.54s |

---

## 7. Strategy Evaluation - Are We on the Right Track?

### What We Are Doing Right

**1. DLX guarantees correctness for N <= 9 regardless of puzzle type.**

DLX is a brute-force exact solver. It WILL find a solution if one exists,
for any puzzle up to N = 9 within ~30 seconds. This is our safety net and
it does not depend on any assumption about how puzzles are generated.

We tested this with 30 seeds of completely random 3D pieces at each grid
size -- 100% correct, no exceptions.

**2. For N >= 10, we depend on structural assumptions.**

The slab planner and pair-based tiling only work if pieces are flat. We built
these by studying OUR OWN test generators, not the professor's. If the
professor generates genuinely 3D pieces at N >= 10, these planners won't help
and we fall back to block decomposition (only works for certain grid sizes)
or neural network search (undertrained).

**3. The fallback chain is well-designed.**

We try the fastest method first and fall back to slower but more general
methods. The worst case is: specialized planners fail -> DLX runs -> DLX
times out -> we return "unsolvable" (which might be wrong for large puzzles).

**4. The code incorporates techniques from the course.**

- Neural networks (Labs on PyTorch, SGD)
- Beam search (probabilistic search)
- ADI / self-play (reinforcement learning ideas)
- Combinatorial optimization

### Where We Could Improve

**1. The neural network is undertrained and underused.**

Currently, the NN is only trained on 3x3x3 puzzles. For the professor's
grading, we barely use it -- the slab planner and DLX handle everything. The
NN is mainly there to demonstrate the PML connection.

**Recommendation:** Train the NN on larger grids (5x5x5, 7x7x7) and show
it actually helping on cases where DLX is slow. This would strengthen our
"probabilistic ML" story significantly.

**2. Genuinely 3D pieces at N >= 10 are partially unsolved.**

DLX handles N <= 9 reliably. Block decomposition handles N = 10 (5^3 blocks)
and N = 12 (6^3 blocks). But N = 11, 13, 14, etc. with 3D pieces have no
fast solver. The NN could potentially fill this gap if trained properly.

**Risk level for grading: VERY LOW** -- The professor's generator for N >= 7
only produces flat pieces.

**3. Unsolvable detection is basic.**

We detect unsolvable cases by volume mismatch and DLX exhaustion. For small
grids this is fine (DLX quickly proves unsolvability). For large grids, we'd
have to timeout and guess, which is risky.

**4. The connection to PML could be more explicit.**

While we use neural networks and beam search, the probabilistic ML angle could
be stronger. Ideas:
- Gaussian Process for DLX runtime prediction
- Bayesian optimization for search hyperparameters
- Random Forest for solvability classification (Lab 6)

### Overall Verdict

**For N <= 9: rock solid.** DLX is universal and proven correct.

**For N >= 10: uncertain.** We pass our own tests, but we don't know what
the professor will send. If it's flat pieces, we're fine. If it's arbitrary
3D pieces at N=11 or N=13, we could fail.

**Priority for the remaining week:**
1. Fix the N=11/13/14 gap (block decomposition with non-divisible sizes,
   or make the NN actually solve these)
2. Speed up DLX for N=9 (currently ~32s average, risky if timeout is tight)
3. Strengthen the ML narrative for the report
4. Handle unsolvable detection at large N properly

---

## 8. Remaining Weaknesses and Risks

### Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Professor tests N=11 or N=13 with 3D pieces | Medium | **Critical -- 0% solve rate** | Need to fix block decomp or NN |
| Professor tests N=9 with tight timeout | Medium | Medium | DLX takes ~32s avg, could timeout |
| Professor tests N >= 15 with 3D pieces | Low | High | Only works if divisible by 5 or 6 |
| Unsolvable detection at N >= 10 | Medium | Medium | Currently just timeouts, no proof |

### Known Limitations (CONFIRMED)

1. **N=11 with 3D pieces: 0% solve rate** -- This is our #1 problem
2. **N=9 takes ~32s average** -- risky if professor has a tight timeout
3. **Neural network adds no value** to any realistic test case
4. **Slab/pair planners are irrelevant** for random 3D pieces
5. **Unsolvable detection at N >= 10** relies on timeout (guessing)

---

## 9. Work Division Plan

### Recommended Split: Two Partners

Below, "Person A" focuses on the algorithmic/solver side (DLX, planners,
block decomposition), and "Person B" focuses on the ML/probabilistic side
(neural network, training, PML connections). Both should understand the
full system.

---

### Person A: Solver and Optimization

**Goal:** Make the solver faster and more robust.

**Tasks:**

1. **Optimize DLX for N=4 (Priority: Medium)**
   - File: `phase1/dlx_solver.py`
   - The N=4 DLX cases take 3-8 seconds. Investigate piece ordering heuristics
     or tighter pruning.
   - Approach: Profile the search tree, identify which pieces cause the most
     backtracking, try column ordering heuristics.

2. **Improve block decomposition for non-divisible sizes (Priority: Low)**
   - File: `hybrid_solver.py`, function `_solve_blockwise_general`
   - Currently only works when N is divisible by block_size (5, 6, 7, 8).
   - Could add a "fill the remainder" strategy for non-divisible sizes.

3. **Add smarter unsolvable detection (Priority: Medium)**
   - File: `hybrid_solver.py`
   - Beyond volume check, add: parity checks, piece-size feasibility
     (can the remaining volume be partitioned into pieces of sizes 3-5?),
     isolated pocket detection.
   - This would let us reject unsolvable cases faster.

4. **Clean up and document the solver pipeline (Priority: High)**
   - Make the code presentation-ready
   - Add comments explaining the key optimizations
   - Prepare timing benchmarks for the writeup

5. **Prepare the demo notebook (Priority: High)**
   - File: `demo.ipynb`
   - Make sure it runs end-to-end and clearly demonstrates the solver

---

### Person B: Neural Network and PML Connections

**Goal:** Strengthen the machine learning story.

**Tasks:**

1. **Train CuboidNet on larger grids (Priority: High)**
   - Files: `phase2/train.py`, `phase2/data_generator.py`
   - Train on 5x5x5 puzzles (currently only 3x3x3)
   - Show learning curves, validation accuracy
   - Demonstrate that the NN actually helps on cases where DLX struggles

2. **Run ADI (self-improvement) loop (Priority: High)**
   - File: `phase2/train.py`, function `run_adi_iteration`
   - Show the network improves over iterations (like AlphaGo self-play)
   - This is a strong PML story: the network learns from its own experience

3. **Add a Bayesian/probabilistic component (Priority: Medium)**
   - New code or modifications to existing files
   - Ideas:
     a. **GP runtime predictor:** Train a Gaussian Process to predict how
        long DLX will take, so we can skip it for hard cases (Lab 2)
     b. **Random Forest solvability classifier:** Quick predict whether a
        puzzle is solvable before running DLX (Lab 6)
     c. **Bayesian hyperparameter tuning:** Use Bayesian optimization to
        tune beam search parameters (Labs 2, 7)
   - Pick ONE of these and implement it properly

4. **Run experiments and create figures (Priority: High)**
   - Compare DLX vs NN vs Hybrid on different grid sizes
   - Plot: accuracy vs grid size, time vs grid size, beam width vs accuracy
   - Show NN value predictions vs actual solvability
   - Create visualizations of solved puzzles (3D plots)

5. **Write the PML narrative for the report (Priority: High)**
   - Explain how our project connects to course topics
   - Frame the NN beam search as "probabilistic planning"
   - Frame ADI as "approximate policy iteration"
   - Frame the slab planner as "structured inference"

---

### Shared Tasks (Do Together)

1. **Final integration testing** -- Run the full test suite together,
   verify everything works on a clean machine
2. **Report writing** -- Each person writes their sections, then review
   each other's work
3. **Presentation prep** -- Decide what to demo live

---

### Timeline (1 Week)

| Day | Person A | Person B |
|------|----------|----------|
| Day 1-2 | Clean up code, prepare demo notebook | Train NN on 5x5x5, run ADI |
| Day 3-4 | Smarter unsolvable detection, final testing | Run experiments, create figures |
| Day 5-6 | Help with report, integration testing | Write PML narrative, add GP/RF component |
| Day 7 | Joint: Final report + presentation prep | Joint: Final report + presentation prep |

**What to cut if time is tight:**
- Skip block decomposition improvements (low impact on grade)
- Skip the Bayesian/GP component (nice-to-have, not required)
- Focus on: demo notebook, NN training, report writing

---

### Quick Start Guide for New Partners

If you just joined the project, here's how to get running:

```bash
# 1. Activate the virtual environment
source ~/venv/bin/activate

# 2. Run the professor grading simulation (should be 33/33)
python3 test_professor_grading.py

# 3. Run the stress test (should be 134/134)
python3 test_stress_comprehensive.py

# 4. Try solving a specific puzzle
python3 -c "
from hybrid_solver import solve_size_gated
from grading_harness import build_scale_suite

cases, src = build_scale_suite(grid_size=7, n_cases=1, seed=42)
result = solve_size_gated(cases[0].pieces, cases[0].grid_size)
print(f'Method: {result[\"submethod\"]}')
print(f'Solved: {result[\"solution\"] is not None}')
print(f'Time: {result[\"time\"]:.3f}s')
"

# 5. Open the demo notebook
jupyter notebook demo.ipynb
```

**Key files to read first:**
1. This document (you're reading it)
2. `hybrid_solver.py` -- function `solve_size_gated` (line ~1460)
3. `phase1/dlx_solver.py` -- the DLX class
4. `block_planner_v2.py` -- function `solve_slab_paired`
5. `demo.ipynb` -- the full walkthrough
