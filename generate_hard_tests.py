"""Generate hard test cases for stress-testing the solver.

Creates cases that probe edge conditions the professor might test:
1. Genuinely 3D pieces at small grids (3-5)
2. Mixed constructive at all grid sizes (7, 9, 12)
3. Striped constructive with various piece-size distributions
4. Volume-mismatch unsolvable cases
5. Cases with unusual piece-size distributions
6. Both absolute and relative (normalized) coordinates

Output: fixtures/hard_test_suite.json
"""
from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

from grading_harness import (
    Case,
    build_constructive_case,
    build_mixed_constructive_case,
    build_striped_constructive_case,
    _normalize_piece_local,
)
from phase1.polycube import normalize
from robust_generator import build_robust_constructive_case


def _make_unsolvable_volume_excess(pieces, grid_size, rng):
    """Add a random small piece to make volume > N³."""
    extra = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    return list(pieces) + [extra]


def _make_unsolvable_volume_drop(pieces, grid_size, rng):
    """Remove a random piece to make volume < N³."""
    if len(pieces) < 2:
        return None
    idx = rng.randrange(len(pieces))
    return [p for i, p in enumerate(pieces) if i != idx]


def _make_unsolvable_shape_mismatch(pieces, grid_size, rng):
    """Swap a piece for a different-shaped piece with the same cell count.
    Volume matches but exact cover becomes impossible (usually)."""
    if len(pieces) < 2:
        return None
    idx = rng.randrange(len(pieces))
    old = pieces[idx]
    size = len(old)
    # Create a straight rod of the same size
    new_piece = [(i, 0, 0) for i in range(size)]
    out = list(pieces)
    out[idx] = new_piece
    return out


def generate_constructive_3d(grid_size, n_cases, seed):
    """Genuinely 3D cases at small grids."""
    rng = random.Random(seed)
    cases = []
    attempts = 0
    while len(cases) < n_cases and attempts < n_cases * 100:
        attempts += 1
        try:
            pieces = build_constructive_case(grid_size, seed=rng.randint(0, 10**9))
            cases.append(Case(
                case_id=f"hard_3d_{grid_size}_{len(cases):02d}",
                grid_size=grid_size,
                pieces=pieces,
                expected_solvable=True,
                stratum=f"{grid_size}^3_solvable_3d",
                generator="constructive_3d",
            ))
        except RuntimeError:
            continue
    return cases


def generate_mixed_variants(grid_size, n_cases, seed, relative=False):
    """Mixed constructive with optional normalization."""
    rng = random.Random(seed)
    cases = []
    for i in range(n_cases):
        pieces = build_mixed_constructive_case(
            grid_size, seed=rng.randint(0, 10**9), relative_pieces=relative,
        )
        tag = "rel" if relative else "abs"
        cases.append(Case(
            case_id=f"hard_mixed_{tag}_{grid_size}_{i:02d}",
            grid_size=grid_size,
            pieces=pieces,
            expected_solvable=True,
            stratum=f"{grid_size}^3_solvable_mixed_{tag}",
            generator=f"mixed_constructive_{tag}",
        ))
    return cases


def generate_striped_variants(grid_size, n_cases, seed, relative=False):
    """Striped constructive with optional normalization."""
    rng = random.Random(seed)
    cases = []
    for i in range(n_cases):
        pieces = build_striped_constructive_case(
            grid_size, seed=rng.randint(0, 10**9), relative_pieces=relative,
        )
        tag = "rel" if relative else "abs"
        cases.append(Case(
            case_id=f"hard_striped_{tag}_{grid_size}_{i:02d}",
            grid_size=grid_size,
            pieces=pieces,
            expected_solvable=True,
            stratum=f"{grid_size}^3_solvable_striped_{tag}",
            generator=f"striped_constructive_{tag}",
        ))
    return cases


def generate_robust_3d(grid_size, n_cases, seed, relative=True):
    """Genuinely 3D pieces via robust_generator, optionally normalized."""
    rng = random.Random(seed)
    cases = []
    for i in range(n_cases):
        s = rng.randint(0, 10**9)
        try:
            pieces = build_robust_constructive_case(grid_size, s)
        except RuntimeError:
            continue
        if relative:
            pieces = [_normalize_piece_local(p) for p in pieces]
        tag = "rel" if relative else "abs"
        cases.append(Case(
            case_id=f"hard_robust3d_{tag}_{grid_size}_{i:02d}",
            grid_size=grid_size,
            pieces=pieces,
            expected_solvable=True,
            stratum=f"{grid_size}^3_solvable_robust3d_{tag}",
            generator=f"robust_3d_{tag}",
        ))
    return cases


def generate_unsolvable(grid_size, n_cases, seed, base_pieces_fn):
    """Unsolvable cases with subtle volume mismatches."""
    rng = random.Random(seed)
    cases = []
    mutators = [
        ("excess", _make_unsolvable_volume_excess),
        ("drop", _make_unsolvable_volume_drop),
    ]
    for i in range(n_cases):
        base = base_pieces_fn(grid_size, rng.randint(0, 10**9))
        mut_name, mut_fn = mutators[i % len(mutators)]
        mutated = mut_fn(base, grid_size, rng)
        if mutated is None:
            continue
        cases.append(Case(
            case_id=f"hard_unsolvable_{mut_name}_{grid_size}_{i:02d}",
            grid_size=grid_size,
            pieces=mutated,
            expected_solvable=False,
            stratum=f"{grid_size}^3_unsolvable",
            generator=f"volume_{mut_name}",
        ))
    return cases


def build_hard_test_suite(seed=12345):
    """Build a comprehensive hard test suite."""
    rng = random.Random(seed)
    all_cases = []

    print("Generating hard test suite...")

    # 1. Genuinely 3D at small grids
    for N in (3, 4):
        cases = generate_constructive_3d(N, 5, rng.randint(0, 10**9))
        print(f"  3D constructive {N}^3: {len(cases)} cases")
        all_cases.extend(cases)

    # 2. Mixed constructive (absolute) at all large sizes
    for N in (7, 9, 12):
        cases = generate_mixed_variants(N, 5, rng.randint(0, 10**9), relative=False)
        print(f"  Mixed abs {N}^3: {len(cases)} cases")
        all_cases.extend(cases)

    # 3. Mixed constructive (relative/normalized) at all large sizes
    for N in (7, 9, 12):
        cases = generate_mixed_variants(N, 3, rng.randint(0, 10**9), relative=True)
        print(f"  Mixed rel {N}^3: {len(cases)} cases")
        all_cases.extend(cases)

    # 4. Striped constructive (absolute)
    for N in (7, 9, 12):
        cases = generate_striped_variants(N, 3, rng.randint(0, 10**9), relative=False)
        print(f"  Striped abs {N}^3: {len(cases)} cases")
        all_cases.extend(cases)

    # 5. Striped constructive (relative/normalized)
    for N in (7, 9, 12):
        cases = generate_striped_variants(N, 3, rng.randint(0, 10**9), relative=True)
        print(f"  Striped rel {N}^3: {len(cases)} cases")
        all_cases.extend(cases)

    # 6. Robust 3D constructive (normalized) at medium/large grids
    for N in (5, 7, 9, 10, 12):
        cases = generate_robust_3d(N, 3, rng.randint(0, 10**9), relative=True)
        print(f"  Robust 3D rel {N}^3: {len(cases)} cases")
        all_cases.extend(cases)

    # 7. Unsolvable at various sizes
    for N in (3, 4, 7, 12):
        def base_fn(gs, s):
            if gs <= 4:
                try:
                    return build_constructive_case(gs, seed=s)
                except RuntimeError:
                    return build_mixed_constructive_case(gs, seed=s)
            return build_mixed_constructive_case(gs, seed=s)

        cases = generate_unsolvable(N, 4, rng.randint(0, 10**9), base_fn)
        print(f"  Unsolvable {N}^3: {len(cases)} cases")
        all_cases.extend(cases)

    print(f"\nTotal: {len(all_cases)} cases")

    # Evaluate quality: check piece-size distribution and shape diversity
    print("\nQuality assessment:")
    for stratum in sorted(set(c.stratum for c in all_cases)):
        stratum_cases = [c for c in all_cases if c.stratum == stratum]
        sizes_all = Counter()
        for c in stratum_cases:
            sizes_all.update(len(p) for p in c.pieces)
        is_3d = any(
            max(max(c2[a] for c2 in p) - min(c2[a] for c2 in p) for a in range(3)) >= 2
            for c in stratum_cases for p in c.pieces
        )
        print(f"  {stratum}: {len(stratum_cases)} cases, "
              f"piece sizes {dict(sizes_all)}, 3D={is_3d}")

    return all_cases


def save_suite(cases, path):
    """Save test suite in the same format as the fixture."""
    data = {
        "version": "hard_test_v1",
        "seed": 12345,
        "n_cases": len(cases),
        "cases": [
            {
                "case_id": c.case_id,
                "grid_size": c.grid_size,
                "pieces": [list(list(int(v) for v in cell) for cell in p) for p in c.pieces],
                "expected_solvable": c.expected_solvable,
                "stratum": c.stratum,
                "generator": c.generator,
            }
            for c in cases
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    cases = build_hard_test_suite()
    save_suite(cases, "fixtures/hard_test_suite.json")
