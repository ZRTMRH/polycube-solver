"""Comprehensive honest validation: relative_pieces=True across many seeds.

This matches the professor's grading conditions more closely than the fixture,
which uses absolute coordinates for case diversity.
"""
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grading_harness import (
    build_mixed_constructive_case,
    build_striped_constructive_case,
)
from hybrid_solver import solve_size_gated

SEEDS_PER_TYPE = 10
TIMEOUT = 60  # seconds per case


def verify_solution(sol, grid_size):
    """Quick check: all cells covered exactly once."""
    all_cells = set()
    for pidx, cells in sol.items():
        for c in cells:
            if c in all_cells:
                return False
            all_cells.add(c)
    expected = {(x, y, z) for x in range(grid_size) for y in range(grid_size) for z in range(grid_size)}
    return all_cells == expected


def main():
    total = 0
    correct = 0
    failures = []

    for N in [7, 9, 12]:
        for case_type, builder in [("mixed", build_mixed_constructive_case),
                                     ("striped", build_striped_constructive_case)]:
            print(f"\n=== N={N} {case_type} (relative_pieces=True) ===")
            sys.stdout.flush()
            for i in range(SEEDS_PER_TYPE):
                seed = 1000 + i * 137 + N
                label = f"{case_type}_rel_{N}_s{seed}"
                total += 1
                try:
                    pieces = builder(N, seed=seed, relative_pieces=True)
                except Exception as e:
                    print(f"  {label}: GEN ERROR ({e})")
                    failures.append(label)
                    continue

                t0 = time.time()
                try:
                    res = solve_size_gated(pieces, N)
                    elapsed = time.time() - t0
                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  {label}: ERROR ({e}) {elapsed:.1f}s")
                    failures.append(label)
                    continue

                if elapsed > TIMEOUT:
                    print(f"  {label}: SLOW {elapsed:.1f}s method={res.get('submethod','?')}")

                if res['solution'] is not None:
                    valid = verify_solution(res['solution'], N)
                    if valid:
                        correct += 1
                        print(f"  {label}: PASS {elapsed:.1f}s method={res.get('submethod','?')}")
                    else:
                        print(f"  {label}: INVALID {elapsed:.1f}s")
                        failures.append(label)
                else:
                    print(f"  {label}: FAIL {elapsed:.1f}s method={res.get('submethod','?')}")
                    failures.append(label)
                sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"HONEST COMPREHENSIVE: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*60}")
    if failures:
        print(f"Failures ({len(failures)}): {failures}")


if __name__ == "__main__":
    main()
