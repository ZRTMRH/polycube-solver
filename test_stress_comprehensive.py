"""Comprehensive stress test across all solver paths and grid sizes."""
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grading_harness import (
    build_mixed_constructive_case,
    build_striped_constructive_case,
    build_scale_suite,
)
from hybrid_solver import solve_size_gated
from phase1.test_cases import verify_solution

SEEDS_PER_TEST = 10
total = 0
correct = 0
max_times = {}


def test(pieces, grid_size, label, expected_solvable=True):
    global total, correct
    total += 1
    t0 = time.time()
    res = solve_size_gated(pieces, grid_size)
    elapsed = time.time() - t0
    solved = res['solution'] is not None
    method = res.get('submethod', '?')

    if solved and expected_solvable:
        valid = verify_solution(res['solution'], grid_size)
        ok = valid
    elif not solved and not expected_solvable:
        ok = True
    else:
        ok = False

    if ok:
        correct += 1

    key = label.rsplit('_s', 1)[0] if '_s' in label else label
    max_times[key] = max(max_times.get(key, 0), elapsed)

    status = "PASS" if ok else "FAIL"
    print(f"  {label}: {status} {elapsed:.3f}s method={method}")
    sys.stdout.flush()
    return ok


# --- Mixed constructive (pair-based solver) ---
print("=== Mixed constructive (relative_pieces=True) ===")
for N in [5, 7, 9, 12, 15, 20]:
    for i in range(SEEDS_PER_TEST):
        seed = 7000 + N * 100 + i
        pieces = build_mixed_constructive_case(N, seed=seed, relative_pieces=True)
        test(pieces, N, f"mixed_{N}_s{seed}")

# --- Striped constructive ---
print("\n=== Striped constructive (relative_pieces=True) ===")
for N in [7, 9, 12, 15]:
    for i in range(SEEDS_PER_TEST):
        seed = 8000 + N * 100 + i
        pieces = build_striped_constructive_case(N, seed=seed, relative_pieces=True)
        test(pieces, N, f"striped_{N}_s{seed}")

# --- Professor-style build_scale_suite ---
print("\n=== Professor build_scale_suite ===")
for N in [3, 4, 5, 7, 9, 12]:
    suite, suite_type = build_scale_suite(N, 5, seed=12345)
    for case in suite:
        test(case.pieces, case.grid_size, f"prof_{case.case_id}")

# --- Single piece (degenerate) ---
print("\n=== Degenerate cases ===")
for N in [2, 3]:
    pieces = [[(x, y, z) for x in range(N) for y in range(N) for z in range(N)]]
    test(pieces, N, f"single_piece_{N}")

# --- Large N push ---
print("\n=== Large N push (mixed, paired solver) ===")
for N in [25, 30]:
    pieces = build_mixed_constructive_case(N, seed=9999, relative_pieces=True)
    test(pieces, N, f"large_mixed_{N}")

# --- Summary ---
print(f"\n{'='*60}")
print(f"STRESS TEST: {correct}/{total}")
print(f"{'='*60}")
print("\nMax times by test group:")
for key in sorted(max_times.keys()):
    print(f"  {key:30s}: {max_times[key]:.3f}s")
