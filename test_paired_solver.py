"""Test the pair-based constructive solver."""
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grading_harness import build_mixed_constructive_case
from block_planner_v2 import solve_slab_paired

SEEDS = 5

for N in [5, 7, 9, 12, 15]:
    print(f"\n{'='*50}")
    print(f"N={N}: testing {SEEDS} seeds")
    print(f"{'='*50}")
    passed = 0
    for si in range(SEEDS):
        seed = 1000 + si * 137 + N
        pieces = build_mixed_constructive_case(N, seed=seed, relative_pieces=True)
        t0 = time.time()
        sol, diag = solve_slab_paired(pieces, N)
        elapsed = time.time() - t0
        if sol:
            all_cells = set()
            for pidx, cells in sol.items():
                all_cells |= cells
            expected = {(x,y,z) for x in range(N) for y in range(N) for z in range(N)}
            valid = all_cells == expected
            if valid:
                passed += 1
                print(f"  seed={seed}: PASS {elapsed:.3f}s retries={diag.get('retries',0)}")
            else:
                print(f"  seed={seed}: INVALID {elapsed:.3f}s")
        else:
            print(f"  seed={seed}: FAIL {elapsed:.3f}s reason={diag.get('reason')}")
        sys.stdout.flush()
    print(f"  Result: {passed}/{SEEDS}")
