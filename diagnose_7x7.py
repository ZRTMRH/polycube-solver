"""Diagnose why 7^3 / 9^3 solvable cases fail in the baseline.

Loads one solvable case of each size from the fixture, runs:
1. Pure DLX with extended timeout (120s) -- is the case trivially solvable?
2. Size-gated with exact_first_max_grid bumped to grid_size.
3. Check mixed_constructive structure: are pieces 5-block concats?
"""
from __future__ import annotations

import time
from collections import Counter

from fixture_a_plus import load_fixture
from hybrid_solver import solve_size_gated
from phase1.solver import solve as dlx_once


def run_one(case, label, **kwargs):
    print(f"\n--- {label} | grid={case.grid_size}^3 | case={case.case_id} ---")
    t0 = time.time()
    result = solve_size_gated(case.pieces, grid_size=case.grid_size, verbose=False, **kwargs)
    elapsed = time.time() - t0
    solved = result.get("solution") is not None if isinstance(result, dict) else result is not None
    sub = result.get("submethod", "?") if isinstance(result, dict) else "?"
    print(f"    result: solved={solved}  submethod={sub}  time={elapsed:.2f}s")
    return solved, elapsed, sub


def main():
    cases = load_fixture()
    by_size = {c.grid_size: [] for c in cases}
    for c in cases:
        by_size[c.grid_size].append(c)

    # Inspect one solvable case at 7 and at 9
    for g in (7, 9):
        solvable = [c for c in by_size[g] if c.expected_solvable]
        if not solvable:
            print(f"No solvable cases at {g}^3"); continue
        case = solvable[0]

        print("=" * 70)
        print(f"GRID {g}^3 — case {case.case_id} ({len(case.pieces)} pieces)")
        print("=" * 70)

        piece_sizes = Counter(len(p) for p in case.pieces)
        print(f"Piece size distribution: {dict(sorted(piece_sizes.items()))}")
        print(f"Total volume: {sum(len(p) for p in case.pieces)} (expected {g**3})")

        # A) Pure DLX w/ 120s
        print("\n(A) Pure DLX, 120s timeout:")
        t0 = time.time()
        try:
            sols = dlx_once(case.pieces, g, find_all=False)
            elapsed = time.time() - t0
            print(f"    DLX: {len(sols)} solution(s), time={elapsed:.2f}s")
        except Exception as e:
            print(f"    DLX crashed: {e}")

        # B) size_gated default
        run_one(case, "(B) size_gated default", timeout_dlx=25, timeout_nn=8)

        # C) size_gated with exact_first_max_grid pushed to grid_size
        run_one(
            case,
            f"(C) size_gated exact_first_max_grid={g}",
            exact_first_max_grid=g,
            exact_first_timeout=60.0,
            timeout_dlx=25,
            timeout_nn=8,
        )


if __name__ == "__main__":
    main()
