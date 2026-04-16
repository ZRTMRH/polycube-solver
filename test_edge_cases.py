"""Edge case tests to find remaining solver weaknesses."""
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grading_harness import (
    build_mixed_constructive_case,
    build_striped_constructive_case,
    _normalize_piece_local,
)
from hybrid_solver import solve_size_gated


def test_and_report(pieces, grid_size, label, expected_solvable=True):
    t0 = time.time()
    res = solve_size_gated(pieces, grid_size)
    elapsed = time.time() - t0
    got = res['solution'] is not None

    if got == expected_solvable:
        if got:
            # Verify
            all_cells = set()
            for pidx, cells in res['solution'].items():
                all_cells |= cells
            expected = {(x,y,z) for x in range(grid_size) for y in range(grid_size) for z in range(grid_size)}
            valid = all_cells == expected
            status = "PASS" if valid else "INVALID"
        else:
            status = "PASS"
    else:
        status = "FAIL"

    method = res.get('submethod', '?')
    print(f"  {label}: {status} {elapsed:.1f}s method={method}")
    sys.stdout.flush()
    return status == "PASS", elapsed


total = 0
correct = 0

# --- Large N mixed with relative_pieces ---
print("=== Large N mixed (relative_pieces=True) ===")
for N in [15]:  # N=15 is 3×5, might be constructible
    try:
        pieces = build_mixed_constructive_case(N, seed=561, relative_pieces=True)
        total += 1
        ok, _ = test_and_report(pieces, N, f"mixed_rel_{N}")
        if ok: correct += 1
    except Exception as e:
        print(f"  mixed_rel_{N}: GEN ERROR ({e})")

# --- Large N striped ---
print("\n=== Large N striped (relative_pieces=True) ===")
for N in [15, 20]:
    try:
        pieces = build_striped_constructive_case(N, seed=561, relative_pieces=True)
        total += 1
        ok, _ = test_and_report(pieces, N, f"striped_rel_{N}")
        if ok: correct += 1
    except Exception as e:
        print(f"  striped_rel_{N}: GEN ERROR ({e})")

# --- N=5 mixed with relative_pieces (flat, small) ---
print("\n=== N=5 mixed (relative_pieces=True) ===")
for seed in [100, 200, 300, 400, 500]:
    try:
        pieces = build_mixed_constructive_case(5, seed=seed, relative_pieces=True)
        total += 1
        ok, _ = test_and_report(pieces, 5, f"mixed_rel_5_s{seed}")
        if ok: correct += 1
    except Exception as e:
        print(f"  N=5 seed={seed}: GEN ERROR ({e})")

# --- Unsolvable: volume correct but pieces don't fit ---
print("\n=== Tricky unsolvable: swap two pieces between solvable cases ===")
for N in [7, 9]:
    p1 = build_mixed_constructive_case(N, seed=100, relative_pieces=True)
    p2 = build_mixed_constructive_case(N, seed=200, relative_pieces=True)
    # Swap first piece from each
    mixed = list(p1)
    mixed[0] = p2[0]
    total += 1
    ok, _ = test_and_report(mixed, N, f"swapped_{N}", expected_solvable=False)
    # Note: this may or may not be solvable — it's hard to predict
    # Just track the result for analysis
    if ok: correct += 1

# --- Stress: many seeds for N=12 ---
print("\n=== N=12 stress test (10 seeds, relative_pieces=True) ===")
max_time = 0
for seed in range(3000, 3010):
    pieces = build_mixed_constructive_case(12, seed=seed, relative_pieces=True)
    total += 1
    ok, elapsed = test_and_report(pieces, 12, f"mixed_rel_12_s{seed}")
    if ok: correct += 1
    max_time = max(max_time, elapsed)
print(f"  Max time: {max_time:.1f}s")

# --- Single piece covering whole grid (degenerate) ---
print("\n=== Degenerate: single piece = whole grid ===")
N = 3
pieces = [[(x,y,z) for x in range(N) for y in range(N) for z in range(N)]]
total += 1
ok, _ = test_and_report(pieces, N, "single_piece_3")
if ok: correct += 1

print(f"\n{'='*60}")
print(f"EDGE CASES: {correct}/{total}")
print(f"{'='*60}")
