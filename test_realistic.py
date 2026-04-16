"""Realistic test: greedy random 3D decomposition (professor-style).
Pieces are arbitrary connected 3D shapes, normalized to origin."""

import random, time, sys, os
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1.polycube import normalize
from hybrid_solver import solve_size_gated
from robust_generator import build_robust_constructive_case


def greedy_decompose(N, seed=0, max_retries=100):
    """Greedily carve random connected pieces of size 3-5 from NxNxN cube."""
    for attempt in range(max_retries):
        rng = random.Random(seed * 1000 + attempt)
        remaining = set(
            (x, y, z) for x in range(N) for y in range(N) for z in range(N)
        )
        pieces = []
        failed = False

        while remaining:
            vol_left = len(remaining)
            if vol_left < 3:
                failed = True
                break

            valid_sizes = []
            for sz in [3, 4, 5]:
                if sz > vol_left:
                    continue
                leftover = vol_left - sz
                if leftover == 0 or leftover >= 3:
                    valid_sizes.append(sz)
            if not valid_sizes:
                failed = True
                break

            target_sz = rng.choice(valid_sizes)

            # Pick a surface cell
            surface = []
            for c in remaining:
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    nb = (c[0]+dx, c[1]+dy, c[2]+dz)
                    if nb not in remaining:
                        surface.append(c)
                        break
            if not surface:
                surface = list(remaining)

            start = rng.choice(surface)
            piece = [start]
            piece_set = {start}
            candidates = []
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                nb = (start[0]+dx, start[1]+dy, start[2]+dz)
                if nb in remaining and nb not in piece_set:
                    candidates.append(nb)

            while len(piece) < target_sz and candidates:
                idx = rng.randrange(len(candidates))
                cell = candidates[idx]
                candidates[idx] = candidates[-1]
                candidates.pop()
                if cell not in remaining or cell in piece_set:
                    continue
                piece.append(cell)
                piece_set.add(cell)
                for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    nb = (cell[0]+dx, cell[1]+dy, cell[2]+dz)
                    if nb in remaining and nb not in piece_set:
                        candidates.append(nb)

            if len(piece) != target_sz:
                failed = True
                break

            new_remaining = remaining - piece_set
            if new_remaining:
                start_check = next(iter(new_remaining))
                visited = {start_check}
                q = deque([start_check])
                while q:
                    c = q.popleft()
                    for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                        nb = (c[0]+dx, c[1]+dy, c[2]+dz)
                        if nb in new_remaining and nb not in visited:
                            visited.add(nb)
                            q.append(nb)
                if len(visited) != len(new_remaining):
                    failed = True
                    break

            remaining = new_remaining
            pieces.append(piece)

        if not failed and not remaining:
            return pieces

    return None


if __name__ == "__main__":
    print("=" * 70)
    print("REALISTIC TEST: Random 3D pieces, normalized (professor-style)")
    print("=" * 70)
    print()

    # Use both generators for coverage
    for N in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        passes = 0
        fails = 0
        gen_fails = 0
        times = []
        methods = {}
        n_seeds = 20 if N <= 7 else (10 if N <= 9 else 5)
        timeout = 30.0 if N <= 7 else (90.0 if N <= 9 else 120.0)

        for seed in range(n_seeds):
            # Try greedy decomposer first, fall back to robust_generator
            pieces_abs = greedy_decompose(N, seed=seed)
            if pieces_abs is None:
                pieces_abs2 = build_robust_constructive_case(N, seed=seed)
                if pieces_abs2 is None:
                    gen_fails += 1
                    continue
                pieces_abs = [list(p) for p in pieces_abs2]

            pieces = [
                list(normalize(frozenset(tuple(c) for c in p)))
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
