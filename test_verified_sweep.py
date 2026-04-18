"""
Full sweep test: generate → rotate → solve → verify for N=3..13.

For each grid size:
  1. Generate pieces with robust_generator (constructive decomposition)
  2. Normalize and randomly rotate each piece (strips position+orientation)
  3. Solve with solve_size_gated
  4. Verify solution with independent verify_solution (index completeness,
     no overlap, full coverage, shape match)
"""

import time
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1.polycube import normalize, rotate, ROTATIONS
from phase1.test_cases import verify_solution
from hybrid_solver import solve_size_gated
from robust_generator import build_robust_constructive_case


def run_sweep():
    print("=" * 74)
    print("VERIFIED SWEEP: generate → rotate → solve → verify  (N=3..13)")
    print("=" * 74)
    print()

    overall_pass = 0
    overall_fail = 0
    overall_skip = 0

    for N in range(3, 14):
        n_seeds = 20 if N <= 7 else (10 if N <= 9 else 5)
        timeout = 30.0 if N <= 7 else (90.0 if N <= 9 else 180.0)

        passes = 0
        solve_fails = 0
        verify_fails = 0
        gen_fails = 0
        times = []
        methods = {}

        for seed in range(n_seeds):
            # --- generate ---
            pieces_abs = build_robust_constructive_case(N, seed=seed)
            if pieces_abs is None:
                gen_fails += 1
                continue

            # --- normalize + random rotate (strips position AND orientation) ---
            rng = random.Random(seed)
            pieces = [
                list(normalize(rotate(p, rng.choice(ROTATIONS))))
                for p in pieces_abs
            ]

            # --- solve ---
            t0 = time.time()
            res = solve_size_gated(pieces, N, verbose=False, timeout_dlx=timeout)
            dt = time.time() - t0
            times.append(dt)

            sol = res.get("solution")
            method = res.get("submethod", res.get("method", "?"))
            methods[method] = methods.get(method, 0) + 1

            if sol is None:
                solve_fails += 1
                if solve_fails <= 2:
                    print(
                        f"  SOLVE-FAIL N={N} seed={seed}: method={method} "
                        f"time={dt:.1f}s {len(pieces)} pcs"
                    )
                continue

            # --- verify with independent checker ---
            valid = verify_solution(sol, N, pieces)
            if valid:
                passes += 1
            else:
                verify_fails += 1
                print(
                    f"  VERIFY-FAIL N={N} seed={seed}: method={method} "
                    f"time={dt:.1f}s {len(pieces)} pcs"
                )

        total_attempted = passes + solve_fails + verify_fails
        avg_t = sum(times) / len(times) if times else 0
        max_t = max(times) if times else 0
        solve_pct = 100 * passes / total_attempted if total_attempted else 0

        if verify_fails > 0:
            status = "INVALID"
        elif solve_pct == 100:
            status = "OK"
        elif solve_pct >= 75:
            status = "WEAK"
        else:
            status = "BROKEN"

        print(
            f"N={N:>2}: {passes}/{total_attempted} solved+verified "
            f"({solve_pct:5.1f}%) [{status:>7}] "
            f"gen_fail={gen_fails} solve_fail={solve_fails} "
            f"verify_fail={verify_fails} "
            f"avg={avg_t:.1f}s max={max_t:.1f}s "
            f"methods={dict(methods)}"
        )
        sys.stdout.flush()

        overall_pass += passes
        overall_fail += solve_fails + verify_fails
        overall_skip += gen_fails

    print()
    print("=" * 74)
    total = overall_pass + overall_fail
    pct = 100 * overall_pass / total if total else 0
    print(
        f"OVERALL: {overall_pass}/{total} solved+verified ({pct:.1f}%) "
        f"skipped={overall_skip}"
    )
    if overall_fail == 0 and overall_pass > 0:
        print("ALL TESTS PASSED AND VERIFIED")
    print("=" * 74)


if __name__ == "__main__":
    run_sweep()
