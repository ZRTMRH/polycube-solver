# 3D Polycube Packing Solver -- Project Guide

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Problem We Are Solving](#2-the-problem-we-are-solving)
3. [File-by-File Explanation](#3-file-by-file-explanation)
4. [How the Solver Works (Big Picture)](#4-how-the-solver-works-big-picture)
5. [Algorithms Explained](#5-algorithms-explained)
6. [Current Performance (Honest Evaluation)](#6-current-performance-honest-evaluation)
7. [Remaining Problems and Priorities](#7-remaining-problems-and-priorities)
8. [Work Division Plan (1 Week)](#8-work-division-plan-1-week)

---

## 1. What Is This Project?

We are building a solver for a 3D puzzle. Given a cubic box of size N x N x N
and a set of small 3D shapes (polycubes, each made of 3-5 unit cubes), our job
is to figure out how to fit all the pieces into the box with no gaps and no
overlaps -- or to correctly determine that it is impossible.

This is for STA 561 (Probabilistic Machine Learning), so we also incorporate
neural-network-based and probabilistic techniques alongside classical
algorithms.

---

## 2. The Problem We Are Solving

**Input:** A grid size N and a list of pieces. Each piece is a set of 3D
coordinates (like Lego blocks made of small cubes), normalized to origin.

**Output:** Either a valid placement (which cell each piece goes into) or
"unsolvable" (if the pieces cannot fill the box exactly).

**Why this is hard:** The number of possible placements grows astronomically
with grid size. Even for a 5x5x5 box (125 cells, ~31 pieces), there are
millions of possible combinations. This is an NP-complete problem -- there
is no known fast algorithm that works for all cases.

**How we expect to be graded:** The professor generates test puzzles of various
sizes. We do not know the professor's generation method, but the most natural
approach is to greedily decompose an N x N x N cube into random connected
pieces of size 3-5 (cutting from the surface, checking connectivity). The
pieces are then normalized (shifted to origin) so we cannot cheat by reading
their original positions.

We need to:
- Correctly solve solvable cases (place all pieces)
- Correctly reject unsolvable cases (say "no solution")
- Do it within a time limit

---

## 3. File-by-File Explanation

### Core Solver Files

#### `hybrid_solver.py` -- The Main Brain (Orchestrator)

This is the central file that decides *which* solving strategy to use for each
puzzle. Think of it as a traffic controller.

**What it does:**
- Receives a puzzle (grid size + pieces)
- Checks basic things first (do the piece volumes add up to the box volume?)
- Picks the best strategy based on the grid size:

| Grid Size | Primary Strategy | Why |
|-----------|-----------------|-----|
| N = 2-4   | DLX exact search | Small enough to try all possibilities |
| N = 5-6   | DLX first, then neural net fallback | Medium -- DLX usually works |
| N = 7-9   | DLX with long timeout | Still works but can be slow (~30s at N=9) |
| N >= 10   | Block decomposition | Split into smaller sub-cubes, solve each |

**Key functions:**
- `solve_size_gated()` -- The main entry point. Routes puzzles by grid size.
- `hybrid_solve()` -- Runs NN beam search + DLX fallback for one puzzle.
- `_solve_blockwise_general()` -- Splits large cubes into sub-cubes.
- `_run_dlx_with_timeout()` -- Runs DLX in a subprocess with hard timeout.

#### `phase1/dlx_solver.py` -- The Exact Search Engine (DLX)

This implements "Algorithm X with Dancing Links" (DLX), invented by Donald
Knuth. It is our most reliable solver -- it is guaranteed to find a solution if
one exists, but can be slow for large puzzles.

**How it works (simplified):**
1. Build a big table where each row represents "place piece X in position Y"
   and each column represents either a cell in the grid or a piece.
2. The goal: pick exactly one row per piece such that every cell column is
   covered exactly once.
3. The algorithm tries placing a piece, then recursively fills the rest. If it
   gets stuck, it backtracks and tries a different placement.
4. The "Dancing Links" trick makes backtracking very fast by using a clever
   linked-list data structure where removing and re-inserting items is O(1).

**Key optimization:** MRV (Minimum Remaining Values) heuristic -- always fill
the cell that has the fewest possible pieces that could go there. This
dramatically cuts dead ends.

**Think of it as:** A very organized trial-and-error search that can undo
mistakes instantly.

#### `phase1/polycube.py` -- Geometry Utilities

Handles the geometry of 3D pieces:
- **Rotation:** Each piece can be rotated 24 different ways in 3D (think of
  all the ways you can orient a die).
- **Normalization:** Slides a piece so its corner is at (0,0,0) for
  comparison.
- **Placement generation:** For each piece, computes every valid position and
  rotation that fits inside the N x N x N grid.

#### `phase1/solver.py` -- DLX Wrapper

A convenience wrapper that takes pieces + grid size, builds the DLX table, and
runs the search. Connects `polycube.py` (geometry) with `dlx_solver.py`
(search).

### Neural Network Files (phase2/)

#### `phase2/nn_model.py` -- The Neural Network Architecture

A 3D convolutional neural network called CuboidNet.

**What it does:** Given a partially-filled box, it predicts:
1. **Value:** "How likely is it that this partial state can lead to a
   complete solution?" (a number from 0 to 1)
2. **Policy:** "Which empty cell should we place the next piece in?"
   (a probability for each cell)

**Architecture:**
- Input: 3D grid with one channel per remaining piece (like layers in an
  image, but 3D)
- Body: residual blocks (similar to image recognition networks -- each block
  learns to refine its input)
- Output: value (single number) and policy (score per cell)

**Think of it as:** An AI that learns from solved puzzles to estimate whether
a partially-built solution is promising.

#### `phase2/nn_solver.py` -- Neural-Network-Guided Beam Search

Uses the neural network to guide a search instead of trying every possibility.

**How beam search works (simplified):**
1. Start with an empty grid.
2. Pick the piece with the fewest valid placements (MRV heuristic).
3. Try placing it in various positions.
4. Ask the neural network: "which of these look most promising?"
5. Keep only the top K most promising partial states (K = beam width).
6. Repeat from step 2 with the next piece.
7. If no beam reaches a complete solution, retry with wider beams.

**Also includes:**
- **Pocket pruning:** If placing a piece creates a tiny isolated empty region
  that no remaining piece can fill, discard that option immediately.
- **Frontier DFS:** If beam search gets close but not all the way, switch to
  depth-first search from the best partial solutions.

#### `phase2/train.py` -- Training the Neural Network

1. Generate solved puzzles (using DLX to find actual solutions).
2. Create training data: positive examples from real solutions, negative
   examples from random placements that lead to dead ends.
3. Train the network to distinguish promising states from dead ends.
4. **ADI (Autodidactic Iteration):** Self-improvement loop where the NN tries
   to solve new puzzles and learns from its successes/failures (similar to
   how AlphaGo improves by playing against itself).

#### `phase2/data_generator.py` -- Training Data Factory

- Enumerates all possible piece shapes of sizes 3-5.
- Creates random puzzles, verifies solvability with DLX.
- Converts puzzle states into tensors the neural network can understand.

### Test and Evaluation Files

#### `grading_harness.py` -- Test Suite Builder

Generates test cases and scores solver correctness. This is code WE wrote
to test our solver -- it is NOT the professor's grading code. We do not know
how the professor generates test cases.

#### `fixture_a_plus.py` -- 150-Case Test Fixture

A large test fixture we built for internal testing. Uses our own generators.

#### `robust_generator.py` -- Random 3D Piece Generator

Generates puzzles by greedy decomposition -- the most realistic test we have.
Starts from the full cube, grows random connected pieces from the surface,
checks connectivity after each cut. Produces fully 3D pieces of arbitrary
shape.

#### `test_realistic.py` -- Realistic Test Script

Tests the solver with random 3D pieces from `robust_generator.py` across
grid sizes N=3 to N=12. This is our most honest benchmark.

#### `demo.ipynb` -- Jupyter Demo Notebook

Demonstrates the full pipeline: DLX solving, NN training, beam search,
hybrid solving.

---

## 4. How the Solver Works (Big Picture)

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
[3] Is N small enough for DLX exact search?
    |-- N <= 4 --> DLX only (guaranteed, usually < 5 sec)
    |-- N = 5-6 --> DLX first (with timeout), then NN fallback
    |-- N = 7-9 --> DLX with long timeout (up to 90 sec)
    v
[4] Is N large and divisible into blocks?
    |-- N divisible by 5 --> Split into 5x5x5 blocks, solve each with DLX
    |-- N divisible by 6 --> Split into 6x6x6 blocks, solve each with DLX
    |-- Neither --> Neural network beam search (currently weak) or timeout
```

---

## 5. Algorithms Explained

### Algorithm 1: DLX (Dancing Links) -- "The Reliable Workhorse"

**Category:** Exact search / constraint satisfaction

Think of a Sudoku solver that tries numbers, backtracks when stuck, and is
guaranteed to find a solution if one exists. DLX is the same idea but for
3D puzzles.

**Strengths:** Guaranteed correct, finds solutions for any puzzle.
**Weaknesses:** Slow for large puzzles (exponential worst case). At N=9 it
takes ~30s per case. At N=11 it times out.
**Used for:** N <= 9 puzzles, solving sub-blocks, training data generation.

### Algorithm 2: Block Decomposition -- "Divide the Box"

**Category:** Divide and conquer

For large grids (N >= 10), split the cube into smaller sub-cubes. For example,
a 10x10x10 grid becomes eight 5x5x5 blocks. Each block is solved independently
with DLX.

**How it works:**
1. Find a block size that divides N evenly (e.g., 5 divides 10).
2. Figure out how many pieces of each size (3, 4, 5) go into each block
   so the volumes add up.
3. Solve each block independently with DLX.
4. Combine the solutions.

**Strengths:** Makes large grids tractable.
**Weaknesses:** Only works when N is divisible by a supported block size (5 or
6 currently). N=11, 13, 14 cannot be decomposed this way.

### Algorithm 3: Neural Network Beam Search -- "Learned Intuition"

**Category:** Deep learning + search
**Course connection:** Direct application of neural networks (Lab PyTorch),
stochastic optimization (Lab 7), and self-play/ADI.

The neural network learns from solved examples to predict which partial
placements are promising. Beam search explores multiple options in parallel,
guided by the network's predictions.

**Strengths:** Can potentially handle cases where DLX is too slow.
**Weaknesses:** Not guaranteed to find a solution; requires training; currently
only trained for 3x3x3 (very undertrained).

### Algorithm 4: Profile Assignment -- "Budget Balancing"

**Category:** Integer constraint satisfaction

When distributing pieces across blocks, we need to decide how many pieces of
each size go where. This is like a budgeting problem: each block needs pieces
that sum to exactly its volume.

**Used in:** Block decomposition.

---

## 6. Current Performance (Honest Evaluation)

Tested with random 3D pieces generated by greedy decomposition (the most
realistic scenario). Pieces are arbitrary connected 3D shapes, normalized to
origin. Three independent runs confirm these results.

| Grid | Pass Rate | Avg Time | Method | Status |
|------|-----------|----------|--------|--------|
| N=3 | 20/20 (100%) | 2-4s | DLX exact | OK |
| N=4 | 20/20 (100%) | 2-4s | DLX exact | OK |
| N=5 | 20/20 (100%) | 3-4s | DLX exact-first | OK |
| N=6 | 20/20 (100%) | 4-6s | DLX exact-first | OK |
| N=7 | 20/20 (100%) | 7-11s | DLX fallback | OK but slow |
| N=8 | 10/10 (100%) | 13-21s | DLX fallback | Slow |
| N=9 | 10/10 (100%) | 31-44s | DLX fallback | Very slow |
| N=10 | 5/5 (100%) | 29-35s | Block decomp (5^3) | OK |
| **N=11** | **0/5 (0%)** | **148-167s** | **DLX timeout** | **BROKEN** |
| N=12 | 5/5 (100%) | 29-80s | Block decomp (6^3) | OK |

**Key takeaways:**
- N <= 9: **DLX handles everything**, guaranteed correct.
- N = 10, 12: **Block decomposition works** (divisible by 5 or 6).
- **N = 11: completely broken** (0% solve rate). Not divisible by 5 or 6,
  and DLX is too slow.
- N = 13, 14, etc. will have the same problem as N = 11.
- The neural network currently adds no value to any realistic test case.

---

## 7. Remaining Problems and Priorities

### Priority 1: Fix the N=11 Gap (CRITICAL)

N=11 is a guaranteed zero. The fix: support **mixed block sizes**. Split
N=11 into blocks of 5 and 6 (5+6=11). This would cover every N >= 10.

Similarly: N=13 = 5+8 or 7+6, N=14 = 7+7 or 8+6, etc.

### Priority 2: Speed Up DLX for N=9 (HIGH)

N=9 takes 31-44 seconds average. If the professor has a strict timeout
(say 30s), we could fail. Options:
- Better piece ordering heuristics
- Tighter pruning in the DLX search
- Use the NN to guide which pieces to try first

### Priority 3: Make the NN Actually Useful (HIGH for PML Story)

The NN is currently trained only on 3x3x3. It needs to be trained on
5x5x5+ to help with cases where DLX struggles. This is also critical
for the PML course connection -- we need to show the NN actually
contributes.

### Priority 4: Unsolvable Detection at Large N (MEDIUM)

For N >= 10, if a puzzle is unsolvable, we currently just timeout and
guess "unsolvable." A trained classifier or better structural checks
would be more reliable.

---

## 8. Work Division Plan (1 Week, 4 People)

### Person A: Block Decomposition (CRITICAL)

**Goal:** Fix the N=11 failure and make all grid sizes work.

1. **Fix N=11 with mixed block sizes (Day 1-3, CRITICAL)**
   - File: `hybrid_solver.py`
   - Currently N=11 is 0% because it's not divisible by 5 or 6.
   - Solution: split into mixed blocks. E.g., 11 = 5+6 along one axis,
     giving blocks of 5x11x11 and 6x11x11, each further subdivided.
   - Or: support "slab" blocks -- e.g., split 11^3 into slabs of thickness
     5 and 6, then solve each slab with DLX.
   - Test: run `python3 test_realistic.py` and check N=11, 13, 14 pass.

2. **Generalize to all N >= 10 (Day 3-5)**
   - Every N >= 10 should be expressible as a sum of 5s and 6s (since
     every integer >= 10 can be written as 5a + 6b for non-negative a, b).
   - Implement and test for N = 11, 13, 14, 16, 17, 19, etc.

### Person B: DLX Optimization

**Goal:** Speed up the brute-force solver for N=7-9.

1. **Profile DLX search at N=9 (Day 1-2)**
   - File: `phase1/dlx_solver.py`
   - N=9 takes 31-44 seconds. Find what causes the most backtracking.
   - Try: better column ordering, piece ordering, symmetry breaking.

2. **Implement improvements (Day 2-5)**
   - Possible: precompute piece compatibility, add constraint propagation,
     improve the MRV heuristic to also consider piece shape constraints.
   - Target: N=9 under 15 seconds, N=8 under 8 seconds.

3. **Update demo notebook (Day 5-6)**
   - File: `demo.ipynb`
   - Make it run end-to-end, demonstrate the full solver pipeline.

### Person C: Neural Network Training

**Goal:** Make the NN actually useful, and demonstrate the PML connection.

1. **Train CuboidNet on 5x5x5 (Day 1-3)**
   - Files: `phase2/train.py`, `phase2/data_generator.py`
   - Generate training data using DLX-solved 5x5x5 puzzles.
   - Show learning curves and validation accuracy.

2. **Run ADI self-improvement loop (Day 3-5)**
   - File: `phase2/train.py`, function `run_adi_iteration`
   - Show the network improves over iterations.
   - This is a key PML story: self-play / approximate policy iteration.

3. **Demonstrate NN value on N=7+ (Day 5-6)**
   - Show that NN-guided beam search can solve cases where pure DLX
     struggles (or is faster).
   - Even partial improvement (e.g., NN solves 60% of N=9 in 5s) is
     valuable for the report.

### Person D: Experiments, Figures, and Report

**Goal:** Create the final report and presentation materials.

1. **Run comprehensive benchmarks (Day 1-3)**
   - Run `test_realistic.py` for all grid sizes, collect timing data.
   - Compare: DLX alone vs Block decomposition vs NN-guided vs Hybrid.
   - Test unsolvable detection (generate unsolvable cases, check accuracy).

2. **Create figures and visualizations (Day 3-5)**
   - Plot: solve time vs grid size, accuracy vs grid size.
   - 3D visualizations of solved puzzles (matplotlib 3D scatter/voxel).
   - NN training curves (loss, accuracy over epochs).
   - Beam search visualization (how the NN narrows down options).

3. **Write the report (Day 4-7)**
   - Frame beam search as "probabilistic planning."
   - Frame ADI as "approximate policy iteration."
   - Frame block decomposition as "divide and conquer with integer
     programming for piece allocation."
   - Connect to course labs (PyTorch, SGD, etc.).

### Everyone: Final Integration (Day 6-7)

- Run the full test suite together.
- Review each other's sections of the report.
- Prepare for presentation / demo.

### Quick Start

```bash
# Activate the virtual environment
source ~/venv/bin/activate

# Run the realistic test (the most honest benchmark)
python3 test_realistic.py

# Try solving a specific puzzle
python3 -c "
from robust_generator import build_robust_constructive_case
from phase1.polycube import normalize
from hybrid_solver import solve_size_gated

pieces = build_robust_constructive_case(7, seed=0)
pieces = [list(normalize(frozenset(p))) for p in pieces]
result = solve_size_gated(pieces, 7, verbose=False)
print(f'Solved: {result[\"solution\"] is not None}')
print(f'Method: {result.get(\"submethod\", result.get(\"method\"))}')
"
```

**Key files to read first:**
1. This document
2. `hybrid_solver.py` -- function `solve_size_gated()`
3. `phase1/dlx_solver.py` -- the DLX class
4. `test_realistic.py` -- the honest benchmark
5. `demo.ipynb` -- the full walkthrough
