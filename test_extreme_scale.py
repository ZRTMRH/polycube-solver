"""Extreme scale test: push paired solver to very large N."""
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grading_harness import build_mixed_constructive_case
from block_planner_v2 import solve_slab_paired
from phase1.test_cases import verify_solution


def test_paired(N, seed=42):
    pieces = build_mixed_constructive_case(N, seed=seed, relative_pieces=True)
    t0 = time.time()
    sol, diag = solve_slab_paired(pieces, N, total_timeout=120)
    elapsed = time.time() - t0
    if sol:
        valid = verify_solution(sol, N)
        print(f"  N={N:3d}: {'PASS' if valid else 'INVALID'} {elapsed:.3f}s  "
              f"pieces={len(pieces):6d}  cells={N**3:7d}")
        return valid
    else:
        print(f"  N={N:3d}: FAIL {elapsed:.3f}s  reason={diag.get('reason')}")
        return False


print("=== Extreme scale: paired solver ===")
total = 0
passed = 0

# Standard range
for N in [5, 7, 9, 12, 15, 20]:
    total += 1
    if test_paired(N):
        passed += 1

# Large even N
for N in [24, 30, 40, 50]:
    total += 1
    if test_paired(N):
        passed += 1

# Large odd N (require rod handling)
for N in [21, 25, 35, 45]:
    total += 1
    if test_paired(N):
        passed += 1

# Multi-seed for N=30
print("\n=== N=30 multi-seed ===")
for seed in range(100, 105):
    total += 1
    if test_paired(30, seed=seed):
        passed += 1

print(f"\n{'='*60}")
print(f"EXTREME SCALE: {passed}/{total}")
print(f"{'='*60}")
