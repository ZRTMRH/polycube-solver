"""Test slab_layered across multiple seeds for N=7,9,12."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grading_harness import build_mixed_constructive_case
from block_planner_v2 import solve_slab_layered

SEEDS = 10

for N in [7, 9, 12]:
    print(f"\n{'='*60}")
    print(f"N={N}: testing {SEEDS} seeds")
    print(f"{'='*60}")
    passed = 0
    total_time = 0
    for si in range(SEEDS):
        seed = 1000 + si * 137 + N
        pieces = build_mixed_constructive_case(N, seed=seed, relative_pieces=True)
        t0 = time.time()
        sol, diag = solve_slab_layered(pieces, N, total_timeout=60.0, max_retries=100)
        elapsed = time.time() - t0
        total_time += elapsed
        if sol:
            # Quick validity check
            all_cells = set()
            for pidx, cells in sol.items():
                all_cells |= cells
            expected = {(x,y,z) for x in range(N) for y in range(N) for z in range(N)}
            valid = all_cells == expected
            if valid:
                passed += 1
                status = "PASS"
            else:
                status = "INVALID"
        else:
            status = f"FAIL({diag.get('reason','')})"
        print(f"  seed={seed}: {status} {elapsed:.2f}s retries={diag.get('retries',0)}")
        sys.stdout.flush()
    print(f"  Result: {passed}/{SEEDS} passed, avg={total_time/SEEDS:.2f}s")
