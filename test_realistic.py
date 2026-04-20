"""Realistic test: greedy random 3D decomposition (professor-style).
Pieces are arbitrary connected 3D shapes, normalized to origin."""

import time, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1.polycube import normalize, rotate, ROTATIONS
from hybrid_solver import solve_size_gated
from robust_generator import build_robust_constructive_case
import random


if __name__ == "__main__":
    print("=" * 70)
    print("REALISTIC TEST: Random 3D pieces, normalized (professor-style)")
    print("=" * 70)
    print()

    for N in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        passes = 0
        fails = 0
        gen_fails = 0
        times = []
        methods = {}
        n_seeds = 20 if N <= 7 else (10 if N <= 9 else 5)
        timeout = 30.0 if N <= 7 else (90.0 if N <= 9 else 120.0)

        for seed in range(n_seeds):
            pieces_abs = build_robust_constructive_case(N, seed=seed)
            if pieces_abs is None:
                gen_fails += 1
                continue

            rng = random.Random(seed)
            pieces = [
                list(normalize(rotate(p, rng.choice(ROTATIONS))))
                for p in pieces_abs
            ]

            t0 = time.time()
            res = solve_size_gated(pieces, N, verbose=False, timeout_dlx=timeout)
            dt = time.time() - t0
            times.append(dt)

            sol = res.get("solution")
            method = res.get("submethod", res.get("method", "?"))
            methods[method] = methods.get(method, 0) + 1

            if sol is not None:
                passes += 1
            else:
                fails += 1
                if fails <= 3:
                    print(
                        f"  FAIL N={N} seed={seed}: method={method} "
                        f"time={dt:.1f}s {len(pieces)} pcs"
                    )

        total = passes + fails
        avg_t = sum(times) / len(times) if times else 0
        max_t = max(times) if times else 0
        pct = 100 * passes / total if total else 0
        status = "OK" if pct == 100 else ("WEAK" if pct >= 75 else "BROKEN")
        print(
            f"N={N:>2}: {passes}/{total} ({pct:5.1f}%) [{status:>6}] "
            f"gen_fail={gen_fails} avg={avg_t:.1f}s max={max_t:.1f}s "
            f"methods={dict(methods)}"
        )
