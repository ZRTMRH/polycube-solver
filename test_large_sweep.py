"""
Large-cube sweep: N=13..25, generate → rotate → solve → verify.

Designed to run within ~16 hours. Adaptive seed counts and wall timeouts.
Logs progress per-seed so progress can be monitored externally.
"""

import time
import sys
import os
import signal
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable per-block logging from hybrid_solver
os.environ["POLYCUBE_LOG_BLOCKS"] = "1"

import logging
_block_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "large_sweep_blocks.log")
_block_handler = logging.FileHandler(_block_log_path, mode="a")
_block_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
logging.getLogger("polycube.blocks").addHandler(_block_handler)
logging.getLogger("polycube.blocks").setLevel(logging.DEBUG)

from phase1.polycube import normalize, rotate, ROTATIONS
from phase1.test_cases import verify_solution
from hybrid_solver import solve_size_gated
from robust_generator import build_robust_constructive_case

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "large_sweep.log")


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def generate_with_timeout(N, seed, timeout):
    """Generate pieces with a wall-clock timeout via SIGALRM."""
    class TimedOut(Exception):
        pass

    def handler(sig, frame):
        raise TimedOut()

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        t0 = time.time()
        pieces = build_robust_constructive_case(N, seed=seed)
        dt = time.time() - t0
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        return pieces, dt
    except TimedOut:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        return None, timeout


def solve_with_timeout(pieces, N, timeout):
    """Solve with a wall-clock timeout via SIGALRM."""
    class TimedOut(Exception):
        pass

    def handler(sig, frame):
        raise TimedOut()

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        t0 = time.time()
        res = solve_size_gated(
            pieces, grid_size=N, verbose=False,
            block_timeout_dlx=30.0,
            block_planner_trials=5,
            block_retries_per_block=5,
        )
        dt = time.time() - t0
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        return res, dt
    except TimedOut:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        return {"solution": None, "submethod": "wall_timeout", "method": None}, timeout


# --- Configuration ---
# (N, num_seeds, gen_timeout_per_seed, solve_timeout_per_seed)
CONFIGS = [
    (13, 5, 600, 600),       # 10 min gen, 10 min solve
    (14, 5, 900, 600),       # 15 min gen, 10 min solve
    (15, 5, 900, 600),
    (16, 3, 1200, 600),      # 20 min gen, 10 min solve
    (17, 3, 1800, 900),      # 30 min gen, 15 min solve
    (18, 3, 1800, 900),
    (19, 3, 2400, 1800),     # 40 min gen, 30 min solve
    (20, 3, 3600, 1800),     # 60 min gen, 30 min solve
    (21, 2, 3600, 1800),     # 60 min gen, 30 min solve
    (22, 2, 5400, 1800),     # 90 min gen, 30 min solve
    (23, 2, 7200, 1800),     # 120 min gen, 30 min solve
    (24, 2, 7200, 2400),     # 120 min gen, 40 min solve
    (25, 2, 7200, 2400),     # 120 min gen, 40 min solve
]


def main():
    sweep_start = time.time()

    log("=" * 78)
    log("LARGE-CUBE SWEEP: N=13..25  generate → rotate → solve → verify")
    log("=" * 78)
    log("")

    # Header
    hdr = (
        f"{'N':>3}  {'seed':>4}  {'gen_s':>8}  {'#pcs':>5}  {'solve_s':>8}  "
        f"{'verify':>6}  {'status':>10}  submethod"
    )
    log(hdr)
    log("-" * len(hdr))

    summary = {}  # N -> {solved, failed, gen_fail, gen_timeout, solve_timeout, verify_fail}

    for N, n_seeds, gen_timeout, solve_timeout in CONFIGS:
        elapsed_total = time.time() - sweep_start
        remaining_h = max(0, (16 * 3600 - elapsed_total) / 3600)
        log(f"--- Starting N={N} ({n_seeds} seeds)  "
            f"[elapsed {elapsed_total/3600:.1f}h, ~{remaining_h:.1f}h remaining] ---")

        stats = {"solved": 0, "failed": 0, "gen_fail": 0,
                 "gen_timeout": 0, "solve_timeout": 0, "verify_fail": 0}

        for seed in range(n_seeds):
            # Check 16h budget
            if time.time() - sweep_start > 16 * 3600:
                log(f"  16h budget exhausted — skipping remaining seeds/sizes")
                break

            # --- Generate ---
            log(f"  N={N} seed={seed}: generating...")
            pieces_abs, gen_time = generate_with_timeout(N, seed, gen_timeout)

            if pieces_abs is None:
                if gen_time >= gen_timeout - 1:
                    status = "GEN_TMOUT"
                    stats["gen_timeout"] += 1
                else:
                    status = "GEN_FAIL"
                    stats["gen_fail"] += 1
                log(f"{N:>3}  {seed:>4}  {gen_time:>8.1f}  {'--':>5}  {'--':>8}  "
                    f"{'--':>6}  {status:>10}  --")
                continue

            # --- Normalize + random rotate ---
            rng = random.Random(seed)
            pieces = [
                list(normalize(rotate(p, rng.choice(ROTATIONS))))
                for p in pieces_abs
            ]

            log(f"  N={N} seed={seed}: generated {len(pieces)} pcs in {gen_time:.1f}s, solving...")

            # --- Solve ---
            res, solve_time = solve_with_timeout(pieces, N, solve_timeout)
            sol = res.get("solution")
            submethod = res.get("submethod", res.get("method", "?"))

            if sol is None:
                if "timeout" in str(submethod):
                    status = "SLV_TMOUT"
                    stats["solve_timeout"] += 1
                else:
                    status = "SLV_FAIL"
                    stats["failed"] += 1
                log(f"{N:>3}  {seed:>4}  {gen_time:>8.1f}  {len(pieces):>5}  "
                    f"{solve_time:>8.1f}  {'--':>6}  {status:>10}  {submethod}")
                continue

            # --- Verify ---
            valid = verify_solution(sol, N, pieces)
            if valid:
                status = "OK"
                stats["solved"] += 1
            else:
                status = "BAD_SOL"
                stats["verify_fail"] += 1

            log(f"{N:>3}  {seed:>4}  {gen_time:>8.1f}  {len(pieces):>5}  "
                f"{solve_time:>8.1f}  {str(valid):>6}  {status:>10}  {submethod}")

        else:
            # Loop completed normally (no break)
            summary[N] = stats
            total = stats["solved"] + stats["failed"] + stats["solve_timeout"] + stats["verify_fail"]
            log(f"  N={N} summary: {stats['solved']}/{total} solved+verified  "
                f"gen_fail={stats['gen_fail']} gen_tmout={stats['gen_timeout']} "
                f"slv_tmout={stats['solve_timeout']} verify_fail={stats['verify_fail']}")
            log("")
            continue
        # break from inner loop → break outer too
        summary[N] = stats
        total = stats["solved"] + stats["failed"] + stats["solve_timeout"] + stats["verify_fail"]
        log(f"  N={N} summary: {stats['solved']}/{total} solved+verified  "
            f"(budget exhausted)")
        break

    # Final summary
    elapsed = time.time() - sweep_start
    log("")
    log("=" * 78)
    log(f"SWEEP COMPLETE  (total time: {elapsed/3600:.2f}h)")
    log("=" * 78)
    log(f"{'N':>3}  {'solved':>6}  {'failed':>6}  {'gen_f':>5}  {'gen_t':>5}  "
        f"{'slv_t':>5}  {'bad':>4}")
    for N in sorted(summary):
        s = summary[N]
        log(f"{N:>3}  {s['solved']:>6}  {s['failed']:>6}  {s['gen_fail']:>5}  "
            f"{s['gen_timeout']:>5}  {s['solve_timeout']:>5}  {s['verify_fail']:>4}")
    log("=" * 78)


if __name__ == "__main__":
    main()
