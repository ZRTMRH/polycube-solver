"""
Stress test: generate NxNxN puzzles (N=12, N=11) with seeds 0-19,
normalize pieces (strip absolute coords), solve with solve_size_gated,
and report pass/fail + submethod + time.

Run with: ~/venv/bin/python3 stress_test_blockwise.py
"""

import sys
import os
import time
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robust_generator import build_robust_constructive_case
from hybrid_solver import solve_size_gated

WALL_TIMEOUT = 120  # seconds per puzzle


def normalize_pieces(pieces):
    """Strip absolute grid coordinates — translate each piece to origin-relative.

    The polycube shape is preserved; only absolute position is removed.
    Each piece is translated so its min (x,y,z) coordinate is (0,0,0).
    This mimics how a solver receives *shape-only* input.
    """
    normalized = []
    for piece in pieces:
        xs = [c[0] for c in piece]
        ys = [c[1] for c in piece]
        zs = [c[2] for c in piece]
        mx, my, mz = min(xs), min(ys), min(zs)
        normalized.append([(x - mx, y - my, z - mz) for x, y, z in piece])
    return normalized


def run_test(N, seeds):
    print(f"\n{'=' * 70}")
    print(f"  N={N}  ({N**3} cells, seeds {seeds[0]}–{seeds[-1]})")
    print(f"{'=' * 70}")
    print(f"{'Seed':>5}  {'Gen(s)':>7}  {'#pcs':>5}  {'Solve(s)':>9}  {'Status':>8}  Submethod")
    print(f"{'-'*5}  {'-'*7}  {'-'*5}  {'-'*9}  {'-'*8}  {'-'*30}")
    sys.stdout.flush()

    results = []

    for seed in seeds:
        # --- Generate puzzle ---
        t_gen = time.time()
        pieces = build_robust_constructive_case(N, seed)
        gen_time = time.time() - t_gen

        if pieces is None:
            print(f"{seed:>5}  {gen_time:>7.2f}  {'?':>5}  {'--':>9}  {'GEN-FAIL':>8}  generator_failed")
            sys.stdout.flush()
            results.append({'seed': seed, 'status': 'gen_failed', 'submethod': None, 'time': None})
            continue

        num_pieces = len(pieces)

        # --- Normalize: strip absolute positions ---
        norm_pieces = normalize_pieces(pieces)

        # --- Solve with wall-clock timeout ---
        t_solve = time.time()

        # Use alarm (UNIX only) as outer hard cap
        timed_out_outer = [False]

        def _alarm_handler(signum, frame):
            timed_out_outer[0] = True
            raise TimeoutError("outer wall clock exceeded")

        try:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(WALL_TIMEOUT + 10)  # +10s buffer over inner timeouts

            res = solve_size_gated(
                norm_pieces,
                grid_size=N,
                model_name=None,        # skip NN, pure planner/DLX path
                block_planner_enabled=True,
                block_timeout_dlx=15.0, # per-block DLX budget
                block_planner_trials=5,
                block_retries_per_block=5,
                timeout_dlx=WALL_TIMEOUT,
                verbose=False,
            )
            signal.alarm(0)  # cancel alarm
        except TimeoutError:
            signal.alarm(0)
            solve_time = time.time() - t_solve
            print(f"{seed:>5}  {gen_time:>7.2f}  {num_pieces:>5}  {solve_time:>9.2f}  {'TIMEOUT':>8}  outer_wall_clock")
            sys.stdout.flush()
            results.append({'seed': seed, 'status': 'timeout', 'submethod': 'outer_wall_clock', 'time': solve_time})
            continue
        except Exception as e:
            signal.alarm(0)
            solve_time = time.time() - t_solve
            print(f"{seed:>5}  {gen_time:>7.2f}  {num_pieces:>5}  {solve_time:>9.2f}  {'ERROR':>8}  {str(e)[:50]}")
            sys.stdout.flush()
            results.append({'seed': seed, 'status': 'error', 'submethod': str(e)[:50], 'time': solve_time})
            continue

        solve_time = time.time() - t_solve

        solved = res.get('solution') is not None
        submethod = res.get('submethod', '??')
        status = 'SOLVED' if solved else 'FAILED'

        print(f"{seed:>5}  {gen_time:>7.2f}  {num_pieces:>5}  {solve_time:>9.2f}  {status:>8}  {submethod}")
        sys.stdout.flush()
        results.append({
            'seed': seed,
            'status': status,
            'submethod': submethod,
            'time': solve_time,
            'num_pieces': num_pieces,
        })

    return results


def summarize(results, label):
    solved = sum(1 for r in results if r.get('status') == 'SOLVED')
    failed = sum(1 for r in results if r.get('status') == 'FAILED')
    timeout = sum(1 for r in results if r.get('status') == 'TIMEOUT')
    error = sum(1 for r in results if r.get('status') in ('error', 'gen_failed'))
    total = len(results)
    times = [r['time'] for r in results if r.get('time') is not None and r.get('status') == 'SOLVED']
    print(f"\n--- Summary [{label}] ---")
    print(f"  Total: {total}  Solved: {solved}  Failed: {failed}  Timeout: {timeout}  Error/GenFail: {error}")
    if times:
        import statistics
        print(f"  Solve times (solved): min={min(times):.2f}s  max={max(times):.2f}s  median={statistics.median(times):.2f}s")
    print(f"  Success rate: {solved}/{total} = {100*solved/total:.1f}%")

    # Breakdown of failure submethods
    fail_methods = {}
    for r in results:
        if r.get('status') != 'SOLVED':
            sm = r.get('submethod', 'unknown')
            fail_methods[sm] = fail_methods.get(sm, 0) + 1
    if fail_methods:
        print(f"  Failure submethods: {fail_methods}")


if __name__ == "__main__":
    seeds = list(range(20))

    print("Blockwise Solver Stress Test")
    print(f"Seeds: {seeds[0]}-{seeds[-1]}  (20 per grid size)")
    print(f"Wall timeout per puzzle: {WALL_TIMEOUT}s")
    print(f"Pieces are NORMALIZED (absolute coords stripped)")

    # N=12: blockwise_6cube (grid_size=12 -> block_sizes=[6])
    results_12 = run_test(12, seeds)
    summarize(results_12, "N=12")

    # N=11: rect_blockwise (grid_size=11 -> split_axis gives [5,6] or similar)
    results_11 = run_test(11, seeds)
    summarize(results_11, "N=11")
