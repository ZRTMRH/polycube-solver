"""Honest baseline test: cases with relative_pieces=True matching professor's grading."""
import time
import sys
import os
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grading_harness import (
    build_mixed_constructive_case,
    build_striped_constructive_case,
    build_constructive_case,
    _normalize_piece_local,
)
from hybrid_solver import solve_size_gated

CASE_TIMEOUT = 120  # seconds per case


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Case timed out")


def run_one(pieces, grid_size, expected_solvable, label):
    """Run one test case with a timeout."""
    sys.stdout.flush()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(CASE_TIMEOUT)
    try:
        t0 = time.time()
        res = solve_size_gated(pieces, grid_size)
        elapsed = time.time() - t0
        signal.alarm(0)

        got_solution = res['solution'] is not None
        ok = got_solution == expected_solvable
        method = res.get('submethod', '?')
        status = "PASS" if ok else "FAIL"
        print(f"  {label}: {status} ({elapsed:.1f}s) method={method}")
        sys.stdout.flush()
        return ok, elapsed, method
    except TimeoutError:
        signal.alarm(0)
        print(f"  {label}: TIMEOUT ({CASE_TIMEOUT}s)")
        sys.stdout.flush()
        return False, CASE_TIMEOUT, 'timeout'
    except Exception as e:
        signal.alarm(0)
        print(f"  {label}: ERROR ({e})")
        sys.stdout.flush()
        return False, 0, 'error'


def main():
    total = 0
    correct = 0
    failures = []

    # --- Small grids: constructive (no normalization needed, naturally 3D) ---
    for N in [3, 4, 5]:
        print(f"\n=== Grid {N}^3 constructive ===")
        sys.stdout.flush()
        for i in range(3):
            try:
                pieces = build_constructive_case(N, seed=561 + N * 10 + i)
            except Exception:
                continue
            total += 1
            ok, elapsed, method = run_one(pieces, N, True, f"constr_{N}_{i}")
            if ok:
                correct += 1
            else:
                failures.append(f"constr_{N}_{i}")

    # --- THE KEY TESTS: large grids with relative_pieces=True ---
    for N in [7, 9, 12]:
        # Mixed constructive (relative)
        print(f"\n=== Grid {N}^3 mixed constructive (relative_pieces=True) ===")
        sys.stdout.flush()
        for i in range(3):
            try:
                pieces = build_mixed_constructive_case(N, seed=561 + N * 10 + i, relative_pieces=True)
            except Exception as e:
                print(f"  mixed_{N}_{i}: GEN ERROR ({e})")
                continue
            total += 1
            ok, elapsed, method = run_one(pieces, N, True, f"mixed_rel_{N}_{i}")
            if ok:
                correct += 1
            else:
                failures.append(f"mixed_rel_{N}_{i}")

        # Striped constructive (relative)
        print(f"\n=== Grid {N}^3 striped constructive (relative_pieces=True) ===")
        sys.stdout.flush()
        for i in range(3):
            try:
                pieces = build_striped_constructive_case(N, seed=561 + N * 10 + i + 100, relative_pieces=True)
            except Exception as e:
                print(f"  striped_{N}_{i}: GEN ERROR ({e})")
                continue
            total += 1
            ok, elapsed, method = run_one(pieces, N, True, f"striped_rel_{N}_{i}")
            if ok:
                correct += 1
            else:
                failures.append(f"striped_rel_{N}_{i}")

    # --- Unsolvable (volume mismatch) - should always work ---
    print(f"\n=== Unsolvable (volume mismatch) ===")
    sys.stdout.flush()
    for N in [3, 7, 12]:
        pieces_base = build_mixed_constructive_case(N, seed=561 + N, relative_pieces=True) if N >= 7 else build_constructive_case(N, seed=561 + N)
        # Drop a piece
        total += 1
        ok, elapsed, method = run_one(pieces_base[1:], N, False, f"unsolvable_{N}")
        if ok:
            correct += 1
        else:
            failures.append(f"unsolvable_{N}")

    print(f"\n{'='*60}")
    print(f"HONEST BASELINE: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*60}")
    if failures:
        print(f"Failures: {failures}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
