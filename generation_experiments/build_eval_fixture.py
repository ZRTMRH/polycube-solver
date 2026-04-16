"""
Build an evaluation fixture spanning grid sizes 4..12, 10 samples each.
Saves to fixtures/eval_4_to_12.json.

Usage:
    python generation_experiments/build_eval_fixture.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import random
from pathlib import Path

from grading_harness import (
    Case,
    _canonical_case_key,
)
from fixture_a_plus import build_unsolvable_from_bases
from robust_generator import build_robust_constructive_case

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "eval_4_to_12.json"
SEED = 42
SAMPLES_PER_GRID = 50
SOLVABLE_FRAC = 0.75

GRID_SIZES = list(range(4, 13))  # 4..12


def _solvable_robust(grid_size, n, seed):
    rng = random.Random(seed)
    cases = []
    seen = set()
    attempts = 0
    while len(cases) < n and attempts < n * 20:
        attempts += 1
        local_seed = rng.randint(0, 10**9)
        pieces = build_robust_constructive_case(grid_size, local_seed)
        key = _canonical_case_key(grid_size, pieces)
        if key in seen:
            continue
        seen.add(key)
        cases.append(Case(
            case_id=f"robust_{grid_size}_{len(cases):02d}",
            grid_size=grid_size,
            pieces=pieces,
            expected_solvable=True,
            stratum=f"{grid_size}^3_solvable",
            generator="robust",
        ))
    if len(cases) < n:
        raise RuntimeError(f"Only built {len(cases)}/{n} robust cases at grid {grid_size}")
    return cases


def build_eval_fixture(seed=SEED, samples_per_grid=SAMPLES_PER_GRID, solvable_frac=SOLVABLE_FRAC):
    rng = random.Random(seed)
    all_cases = []
    n_solvable = round(samples_per_grid * solvable_frac)
    n_unsolvable = samples_per_grid - n_solvable

    for grid_size in GRID_SIZES:
        print(f"\nGrid {grid_size}^3: {n_solvable} solvable, {n_unsolvable} unsolvable")

        sub_seed = rng.randint(0, 10**9)
        solvable = _solvable_robust(grid_size, n_solvable, sub_seed)

        for i, c in enumerate(solvable):
            c.case_id = f"eval_solv_{grid_size}_{i:02d}"

        print(f"  Built {len(solvable)} solvable cases")

        unsub_seed = rng.randint(0, 10**9)
        unsolvable = build_unsolvable_from_bases(grid_size, n_unsolvable, solvable, unsub_seed)
        for i, c in enumerate(unsolvable):
            c.case_id = f"eval_unsolv_{grid_size}_{i:02d}"
            c.stratum = f"{grid_size}^3_unsolvable"

        print(f"  Built {len(unsolvable)} unsolvable cases")

        all_cases.extend(solvable)
        all_cases.extend(unsolvable)

    return all_cases


def _case_to_dict(c):
    return {
        "case_id": c.case_id,
        "grid_size": c.grid_size,
        "pieces": [[list(cell) for cell in piece] for piece in c.pieces],
        "expected_solvable": c.expected_solvable,
        "stratum": c.stratum,
        "generator": c.generator,
    }


def save(cases, path=OUTPUT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "seed": SEED,
        "n_cases": len(cases),
        "grid_sizes": GRID_SIZES,
        "samples_per_grid": SAMPLES_PER_GRID,
        "solvable_ratio": f"{sum(1 for c in cases if c.expected_solvable)}/{len(cases)}",
        "cases": [_case_to_dict(c) for c in cases],
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {len(cases)} cases to {path}")
    return path


if __name__ == "__main__":
    print(f"Building eval fixture: grids {GRID_SIZES[0]}..{GRID_SIZES[-1]}, "
          f"{SAMPLES_PER_GRID} samples each, seed={SEED}")

    cases = build_eval_fixture(samples_per_grid=SAMPLES_PER_GRID, solvable_frac=SOLVABLE_FRAC)

    n_solv = sum(1 for c in cases if c.expected_solvable)
    n_unsolv = len(cases) - n_solv
    print(f"\nTotal: {len(cases)} cases  ({n_solv} solvable / {n_unsolv} unsolvable = "
          f"{n_solv/len(cases):.0%} / {n_unsolv/len(cases):.0%})")

    save(cases)
