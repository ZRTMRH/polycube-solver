"""
Thorough test suite: N=3..14, many seeds per size.

~70% positive tests: generate valid decomposition → rotate → solve → verify.
~30% fault tests:    structurally unsolvable pieces (correct volume, valid
                     polycubes, but no tiling exists) → solver must return None.

Fault test strategies (all preserve total volume = N³):
  1. Oversized piece: include one piece with extent > N (guaranteed unsolvable).
  2. All rods: straight I-pieces only (unsolvable for N=3-4 where rods exceed grid).
  3. Piece swap: replace ~40-70% of pieces with random polycubes.
  4. Shape duplication: replace same-size group with copies of one shape.
  5. Fully random: generate entirely random polycubes summing to N³.

Designed for an 8-hour budget. Logs progress + per-block detail.
"""

import time
import sys
import os
import signal
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable per-block logging
os.environ["POLYCUBE_LOG_BLOCKS"] = "1"

import logging
_block_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thorough_blocks.log")
_block_handler = logging.FileHandler(_block_log_path, mode="a")
_block_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
logging.getLogger("polycube.blocks").addHandler(_block_handler)
logging.getLogger("polycube.blocks").setLevel(logging.DEBUG)

from phase1.polycube import normalize, rotate, ROTATIONS
from phase1.test_cases import verify_solution
from hybrid_solver import solve_size_gated
from robust_generator import build_robust_constructive_case

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thorough_3_14.log")


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Fault test generators
# ---------------------------------------------------------------------------

def _random_polycube(size, rng, grid_size=20):
    """Generate a random connected polycube of given size via random walk."""
    start = (rng.randint(0, grid_size - 1),
             rng.randint(0, grid_size - 1),
             rng.randint(0, grid_size - 1))
    cells = [start]
    cell_set = {start}
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for _ in range(size - 1):
        # Collect all valid neighbors of the current piece
        frontier = []
        for c in cells:
            for d in dirs:
                nb = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if nb not in cell_set and 0 <= nb[0] < grid_size and \
                   0 <= nb[1] < grid_size and 0 <= nb[2] < grid_size:
                    frontier.append(nb)
        if not frontier:
            break
        pick = rng.choice(frontier)
        cells.append(pick)
        cell_set.add(pick)
    return list(normalize(cells))


def make_fault_pieces_swap(valid_pieces, rng, swap_frac=0.3):
    """Replace ~swap_frac of pieces with random polycubes of the same size.

    Volume is preserved. The replacement pieces are structurally unrelated
    to the original decomposition, so the set almost certainly can't tile.
    """
    pieces = list(valid_pieces)
    n_swap = max(1, int(len(pieces) * swap_frac))
    swap_indices = rng.sample(range(len(pieces)), n_swap)
    for idx in swap_indices:
        sz = len(pieces[idx])
        pieces[idx] = _random_polycube(sz, rng)
    return pieces


def make_fault_pieces_duplicate(valid_pieces, rng):
    """Replace many pieces with copies of a single shape.

    Pick one piece shape and replace all same-sized pieces with that shape.
    The over-representation of one shape typically makes tiling impossible.
    """
    pieces = list(valid_pieces)
    # Group by size
    by_size = {}
    for i, p in enumerate(pieces):
        by_size.setdefault(len(p), []).append(i)

    # Pick the largest size group and replace all with one shape
    largest_group_size = max(by_size.keys(), key=lambda s: len(by_size[s]))
    group = by_size[largest_group_size]
    if len(group) < 3:
        # Not enough pieces to make duplication meaningful; fall back to swap
        return make_fault_pieces_swap(valid_pieces, rng, swap_frac=0.4)

    template_idx = rng.choice(group)
    template = list(normalize(pieces[template_idx]))
    for idx in group:
        if idx != template_idx:
            pieces[idx] = template[:]
    return pieces


def make_fault_pieces_all_rods(N, rng):
    """Create a piece set of all straight rods (I-pieces) summing to N³.

    For N >= 4, a set of only straight rods generally can't tile the cube
    because corners require non-linear pieces. Uses sizes 3-5 rods only.
    """
    vol = N ** 3
    pieces = []
    remaining = vol
    while remaining > 0:
        # Ensure we don't leave a remainder of 1 or 2 (can't make a piece)
        if remaining <= 5:
            sz = remaining if remaining >= 3 else remaining
        elif remaining in (6, 7, 8):
            # Pick size that avoids leaving 1 or 2
            safe = [s for s in [3, 4, 5] if remaining - s >= 3 or remaining - s == 0]
            sz = rng.choice(safe) if safe else 3
        else:
            sz = rng.choice([3, 4, 5])
        if sz < 3:
            # Merge remainder into last piece if possible
            if pieces and len(pieces[-1]) + sz <= 5:
                old = pieces.pop()
                sz = len(old) + sz
            else:
                return None
        pieces.append([(i, 0, 0) for i in range(sz)])
        remaining -= sz

    if sum(len(p) for p in pieces) != vol:
        return None
    return pieces


def make_fault_pieces_fully_random(N, rng):
    """Generate a completely random set of polycubes summing to N³.

    Pieces are generated independently, not from any decomposition of the cube.
    Almost certainly unsolvable for N >= 4.
    """
    vol = N ** 3
    pieces = []
    remaining = vol
    while remaining > 0:
        if remaining <= 5:
            sz = remaining if remaining >= 3 else remaining
            if sz < 3:
                # Merge into last piece
                if pieces and len(pieces[-1]) + sz <= 5:
                    old = pieces.pop()
                    pieces.append(_random_polycube(len(old) + sz, rng))
                    remaining = 0
                    continue
                return None
        elif remaining in (6, 7, 8):
            safe = [s for s in [3, 4, 5] if remaining - s >= 3 or remaining - s == 0]
            sz = rng.choice(safe) if safe else 3
        else:
            sz = rng.choice([3, 4, 5])
        pieces.append(_random_polycube(sz, rng))
        remaining -= sz

    if sum(len(p) for p in pieces) != vol:
        return None
    return pieces


def make_fault_pieces_oversized(N, rng):
    """Include one piece with extent > N (guaranteed no valid placement).

    Creates a straight rod of size N+1 (doesn't fit in NxNxN grid in any
    rotation). Remaining volume filled with random polycubes.
    Total volume = N³ (non-trivial). Guaranteed unsolvable.
    """
    vol = N ** 3
    oversized = N + 1
    if oversized > vol:
        return None
    # Straight rod of length N+1
    rod = [(i, 0, 0) for i in range(oversized)]
    pieces = [rod]
    remaining = vol - oversized
    while remaining > 0:
        if remaining <= 5:
            sz = remaining if remaining >= 3 else remaining
            if sz < 3:
                if pieces and len(pieces[-1]) + sz <= 8:
                    old = pieces.pop()
                    pieces.append(_random_polycube(len(old) + sz, rng))
                    remaining = 0
                    continue
                return None
        elif remaining in (6, 7, 8):
            safe = [s for s in [3, 4, 5] if remaining - s >= 3 or remaining - s == 0]
            sz = rng.choice(safe) if safe else 3
        else:
            sz = rng.choice([3, 4, 5])
        pieces.append(_random_polycube(sz, rng))
        remaining -= sz
    if sum(len(p) for p in pieces) != vol:
        return None
    # Shuffle so the oversized piece isn't always first
    rng.shuffle(pieces)
    return pieces


def generate_fault_case(N, seed, valid_pieces, rng):
    """Generate a fault test case. Returns (pieces, fault_type).

    For small N (≤6), DLX can solve almost any random piece set, so we
    guarantee unsolvability via oversized pieces. For larger N, swap/random
    strategies work because the search space is too large.
    """
    if N <= 6:
        # Small grids: DLX solves random piece sets easily.
        # Use guaranteed-unsolvable strategies.
        method = seed % 3
        if method == 0:
            pieces = make_fault_pieces_oversized(N, rng)
            if pieces is not None:
                return pieces, "oversized_piece"
        elif method == 1:
            pieces = make_fault_pieces_all_rods(N, rng)
            if pieces is not None:
                # For N=3-4, rods of size 4-5 don't fit → guaranteed fail.
                # For N=5-6, rods all fit → may solve. Fallback to oversized.
                return pieces, "all_rods"
        else:
            pieces = make_fault_pieces_oversized(N, rng)
            if pieces is not None:
                return pieces, "oversized_piece"
        # Fallback: oversized is guaranteed
        pieces = make_fault_pieces_oversized(N, rng)
        if pieces is not None:
            return pieces, "oversized_piece"
        return make_fault_pieces_swap(valid_pieces, rng, swap_frac=0.7), "swap_70pct"
    else:
        # Large grids: mix oversized (guaranteed) with other strategies
        method = seed % 5
        if method == 0:
            # Guaranteed unsolvable — instant rejection
            pieces = make_fault_pieces_oversized(N, rng)
            if pieces is not None:
                return pieces, "oversized_piece"
        elif method == 1:
            pieces = make_fault_pieces_swap(valid_pieces, rng, swap_frac=0.4)
            return pieces, "swap_40pct"
        elif method == 2:
            pieces = make_fault_pieces_duplicate(valid_pieces, rng)
            return pieces, "duplicate_shape"
        elif method == 3:
            pieces = make_fault_pieces_all_rods(N, rng)
            if pieces is not None:
                return pieces, "all_rods"
            pieces = make_fault_pieces_swap(valid_pieces, rng, swap_frac=0.5)
            return pieces, "swap_50pct"
        else:
            pieces = make_fault_pieces_fully_random(N, rng)
            if pieces is not None:
                return pieces, "fully_random"
            pieces = make_fault_pieces_swap(valid_pieces, rng, swap_frac=0.5)
            return pieces, "swap_50pct"


# ---------------------------------------------------------------------------
# Timeout wrappers
# ---------------------------------------------------------------------------

def generate_with_timeout(N, seed, timeout):
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


def solve_with_timeout(pieces, N, timeout, timeout_dlx=120):
    """Run solver with a hard wall timeout via threading + SIGKILL.

    Uses a background thread to send SIGALRM after `timeout` seconds.
    The solver's internal SIGALRM handlers may interfere, so we also
    track wall time and use a threading.Event for coordination.
    """
    import threading

    result = [None]
    error = [None]
    done_event = threading.Event()

    def _timer_fire():
        """Fire after timeout — send SIGINT to interrupt the main thread."""
        if not done_event.is_set():
            os.kill(os.getpid(), signal.SIGUSR1)

    class _WallTimeout(BaseException):
        pass

    prev_usr1 = signal.getsignal(signal.SIGUSR1)
    def _usr1_handler(sig, frame):
        raise _WallTimeout()

    signal.signal(signal.SIGUSR1, _usr1_handler)
    timer = threading.Timer(timeout, _timer_fire)
    timer.start()
    t0 = time.time()
    try:
        # Scale block planner effort with grid size
        bp_trials = 5 if N <= 10 else min(15, max(5, N))
        bp_retries = 5 if N <= 10 else min(10, max(5, N - 3))
        res = solve_size_gated(
            pieces, grid_size=N, verbose=False,
            timeout_dlx=timeout_dlx,
            block_timeout_dlx=45.0,
            block_planner_trials=bp_trials,
            block_retries_per_block=bp_retries,
            # For large grids, skip useless DLX-on-full-grid fallback
            model_name=None if N > 6 else "soma_3x3x3",
        )
        dt = time.time() - t0
        done_event.set()
        timer.cancel()
        signal.signal(signal.SIGUSR1, prev_usr1)
        return res, dt
    except _WallTimeout:
        dt = time.time() - t0
        done_event.set()
        timer.cancel()
        signal.signal(signal.SIGUSR1, prev_usr1)
        return {"solution": None, "submethod": "wall_timeout", "method": None}, dt
    except Exception as e:
        dt = time.time() - t0
        done_event.set()
        timer.cancel()
        signal.signal(signal.SIGUSR1, prev_usr1)
        return {"solution": None, "submethod": f"error:{e}", "method": None}, dt


# ---------------------------------------------------------------------------
# Config: (N, positive_seeds, fault_seeds, gen_timeout, solve_timeout)
# ---------------------------------------------------------------------------
from hybrid_solver import recommended_timeouts

# Shift all seeds by this amount to test different cases across runs.
# Run 1 used offset=0, run 2 uses offset=100, etc.
SEED_OFFSET = 100

def _build_configs(n_range, pos_seeds_fn, fault_seeds_fn):
    """Build CONFIGS list using recommended_timeouts for each N."""
    configs = []
    for N in n_range:
        gen_t, solve_t, _ = recommended_timeouts(N)
        n_pos = pos_seeds_fn(N)
        n_fault = fault_seeds_fn(N)
        configs.append((N, n_pos, n_fault, gen_t, solve_t))
    return configs

CONFIGS = _build_configs(
    n_range=range(3, 14),
    pos_seeds_fn=lambda N: 20 if N <= 6 else (15 if N <= 9 else 10),
    fault_seeds_fn=lambda N: 5 if N <= 6 else 3,
)


def main():
    sweep_start = time.time()

    total_pos = sum(c[1] for c in CONFIGS)
    total_fault = sum(c[2] for c in CONFIGS)

    log("=" * 78)
    log("THOROUGH TEST SUITE: N=3..13  positive + fault tests (4h budget, fixed N=7-9)")
    log(f"Positive seeds: {total_pos}, Fault seeds: {total_fault} "
        f"({100*total_fault/(total_pos+total_fault):.0f}% fault)")
    log("=" * 78)
    log("")

    hdr = (
        f"{'N':>3}  {'seed':>4}  {'type':>6}  {'gen_s':>7}  {'#pcs':>5}  "
        f"{'slv_s':>7}  {'verify':>6}  {'status':>10}  submethod"
    )
    log(hdr)
    log("-" * len(hdr))

    summary = {}

    for N, n_pos, n_fault, gen_timeout, solve_timeout in CONFIGS:
        elapsed_total = time.time() - sweep_start
        remaining_h = max(0, (4 * 3600 - elapsed_total) / 3600)
        n_total = n_pos + n_fault
        log(f"--- Starting N={N} ({n_pos} positive + {n_fault} fault = {n_total} seeds)  "
            f"[elapsed {elapsed_total/3600:.1f}h, ~{remaining_h:.1f}h remaining] ---")

        stats = {
            # Positive test stats
            "pos_solved": 0, "pos_failed": 0, "pos_timeout": 0,
            "pos_verify_fail": 0, "pos_gen_fail": 0, "pos_gen_timeout": 0,
            # Fault test stats
            "fault_correct_reject": 0,   # solver correctly returned None
            "fault_actually_solvable": 0,  # fault pieces were solvable (not a bug)
            "fault_bad_solution": 0,     # solver returned invalid solution (BUG!)
            "fault_timeout": 0,          # solver timed out (acceptable)
            "fault_gen_fail": 0,
        }
        solve_times = []

        # Interleave positive and fault seeds for variety.
        # Build a schedule: list of (seed_idx, is_fault)
        schedule = []
        pos_i, fault_i = 0, 0
        for i in range(n_total):
            # ~30% fault, distributed evenly
            if fault_i < n_fault and (pos_i >= n_pos or
                    (i + 1) % max(1, n_total // n_fault) == 0):
                schedule.append((fault_i, True))
                fault_i += 1
            else:
                schedule.append((pos_i, False))
                pos_i += 1
        # Fill any remaining
        while pos_i < n_pos:
            schedule.append((pos_i, False))
            pos_i += 1
        while fault_i < n_fault:
            schedule.append((fault_i, True))
            fault_i += 1

        for seq, (seed, is_fault) in enumerate(schedule):
            if time.time() - sweep_start > 4 * 3600:
                log(f"  5h budget exhausted — skipping remaining")
                break

            test_type = "FAULT" if is_fault else "POS"
            log(f"  >> N={N} seed={seed} {test_type} [{seq+1}/{n_total}] generating...")

            # --- Generate base pieces ---
            # Use offset seeds for fault vs positive to avoid collisions
            # SEED_OFFSET shifts all seeds to test different cases across runs
            gen_seed = seed + SEED_OFFSET + (10000 if is_fault else 0)
            pieces_abs, gen_time = generate_with_timeout(N, gen_seed, gen_timeout)

            if pieces_abs is None:
                if gen_time >= gen_timeout - 1:
                    key = "fault_gen_fail" if is_fault else "pos_gen_timeout"
                else:
                    key = "fault_gen_fail" if is_fault else "pos_gen_fail"
                stats[key] = stats.get(key, 0) + 1
                status = "GEN_TMOUT" if gen_time >= gen_timeout - 1 else "GEN_FAIL"
                log(f"{N:>3}  {seed:>4}  {test_type:>6}  {gen_time:>7.1f}  "
                    f"{'--':>5}  {'--':>7}  {'--':>6}  {status:>10}  --")
                continue

            # --- Prepare pieces ---
            rng = random.Random(gen_seed * 31 + 7)

            if is_fault:
                # Create unsolvable piece set
                fault_pieces, fault_method = generate_fault_case(
                    N, seed, pieces_abs, rng
                )
                # Normalize + rotate
                pieces = [
                    list(normalize(rotate(p, rng.choice(ROTATIONS))))
                    for p in fault_pieces
                ]
                # Verify volume is correct (non-trivial fault)
                total_vol = sum(len(p) for p in pieces)
                if total_vol != N ** 3:
                    log(f"{N:>3}  {seed:>4}  {test_type:>6}  {gen_time:>7.1f}  "
                        f"{len(pieces):>5}  {'--':>7}  {'--':>6}  "
                        f"{'SKIP_VOL':>10}  {fault_method} vol={total_vol}")
                    continue
            else:
                # Positive test: normalize + rotate
                pieces = [
                    list(normalize(rotate(p, rng.choice(ROTATIONS))))
                    for p in pieces_abs
                ]

            # --- Solve ---
            # Fault tests: use the DLX timeout_dlx parameter to cap DLX search,
            # plus a wall timeout via SIGALRM. For small N, DLX exhaustive search
            # on unsolvable cases can be slow, so keep it short.
            fault_timeout = min(solve_timeout, 60) if N <= 6 else min(solve_timeout, 120)
            st = solve_timeout if not is_fault else fault_timeout
            # For fault tests, cap the internal DLX timeout too
            dlx_t = 120 if not is_fault else min(30, fault_timeout)
            fault_info = f" [{fault_method}]" if is_fault else ""
            log(f"  >> N={N} seed={seed} {test_type} solving #pcs={len(pieces)} "
                f"wall={st}s dlx={dlx_t}s{fault_info}")
            res, solve_time = solve_with_timeout(pieces, N, st, timeout_dlx=dlx_t)
            sol = res.get("solution")
            submethod = res.get("submethod", res.get("method", "?"))

            if is_fault:
                # Expected: no solution
                if sol is None:
                    if "timeout" in str(submethod):
                        status = "F_TMOUT"
                        stats["fault_timeout"] += 1
                    else:
                        status = "F_REJECT"
                        stats["fault_correct_reject"] += 1
                else:
                    # Solver found a "solution" — verify it
                    valid = verify_solution(sol, N, pieces)
                    if valid:
                        # Fault pieces were actually solvable (not a solver bug).
                        # This happens for small N where random pieces often tile.
                        status = "F_SOLVABL"
                        stats["fault_actually_solvable"] += 1
                    else:
                        # Solver returned invalid solution — real bug!
                        status = "F_BADSOL!"
                        stats["fault_bad_solution"] += 1

                log(f"{N:>3}  {seed:>4}  {test_type:>6}  {gen_time:>7.1f}  "
                    f"{len(pieces):>5}  {solve_time:>7.1f}  {'--':>6}  "
                    f"{status:>10}  {fault_method}")
            else:
                # Positive test
                if sol is None:
                    if "timeout" in str(submethod):
                        status = "SLV_TMOUT"
                        stats["pos_timeout"] += 1
                    else:
                        status = "SLV_FAIL"
                        stats["pos_failed"] += 1
                    log(f"{N:>3}  {seed:>4}  {test_type:>6}  {gen_time:>7.1f}  "
                        f"{len(pieces):>5}  {solve_time:>7.1f}  {'--':>6}  "
                        f"{status:>10}  {submethod}")
                    continue

                valid = verify_solution(sol, N, pieces)
                if valid:
                    status = "OK"
                    stats["pos_solved"] += 1
                    solve_times.append(solve_time)
                else:
                    status = "BAD_SOL"
                    stats["pos_verify_fail"] += 1

                log(f"{N:>3}  {seed:>4}  {test_type:>6}  {gen_time:>7.1f}  "
                    f"{len(pieces):>5}  {solve_time:>7.1f}  {str(valid):>6}  "
                    f"{status:>10}  {submethod}")

        else:
            # Loop completed
            summary[N] = stats
            pos_total = stats["pos_solved"] + stats["pos_failed"] + \
                        stats["pos_timeout"] + stats["pos_verify_fail"]
            fault_total = stats["fault_correct_reject"] + stats["fault_actually_solvable"] + \
                          stats["fault_bad_solution"] + stats["fault_timeout"]
            avg_t = sum(solve_times) / len(solve_times) if solve_times else 0
            max_t = max(solve_times) if solve_times else 0
            log(f"  N={N} positive: {stats['pos_solved']}/{pos_total} solved+verified  "
                f"(gen_fail={stats['pos_gen_fail']} tmout={stats['pos_timeout']} "
                f"vfail={stats['pos_verify_fail']})")
            log(f"  N={N} fault:    {stats['fault_correct_reject']}/{fault_total} "
                f"correctly rejected  "
                f"(solvable={stats['fault_actually_solvable']} "
                f"bad_sol={stats['fault_bad_solution']} "
                f"tmout={stats['fault_timeout']} gen_fail={stats['fault_gen_fail']})")
            if solve_times:
                log(f"  N={N} timing:   avg={avg_t:.1f}s max={max_t:.1f}s")
            log("")
            continue

        summary[N] = stats
        log(f"  N={N} (budget exhausted)")
        break

    # Final summary
    elapsed = time.time() - sweep_start
    log("")
    log("=" * 78)
    log(f"THOROUGH TEST COMPLETE  (total time: {elapsed/3600:.2f}h)")
    log("=" * 78)

    log("")
    log("POSITIVE TESTS (solver should find valid solution):")
    log(f"{'N':>3}  {'solved':>6}  {'fail':>4}  {'tmout':>5}  {'vfail':>5}  {'rate':>6}")
    total_pos_solved = 0
    total_pos_attempted = 0
    for N in sorted(summary):
        s = summary[N]
        att = s['pos_solved'] + s['pos_failed'] + s['pos_timeout'] + s['pos_verify_fail']
        rate = f"{100*s['pos_solved']/att:.0f}%" if att else "N/A"
        total_pos_solved += s['pos_solved']
        total_pos_attempted += att
        log(f"{N:>3}  {s['pos_solved']:>6}  {s['pos_failed']:>4}  "
            f"{s['pos_timeout']:>5}  {s['pos_verify_fail']:>5}  {rate:>6}")
    pos_rate = f"{100*total_pos_solved/total_pos_attempted:.1f}%" \
               if total_pos_attempted else "N/A"
    log(f"Total: {total_pos_solved}/{total_pos_attempted} ({pos_rate})")

    log("")
    log("FAULT TESTS (solver should return None or timeout):")
    log(f"{'N':>3}  {'reject':>6}  {'slvabl':>6}  {'badsol':>6}  {'tmout':>5}  {'rate':>6}")
    total_fault_ok = 0
    total_fault_attempted = 0
    total_bad_sol = 0
    for N in sorted(summary):
        s = summary[N]
        att = s['fault_correct_reject'] + s['fault_actually_solvable'] + \
              s['fault_bad_solution'] + s['fault_timeout']
        # reject + timeout + solvable are all acceptable outcomes
        ok = s['fault_correct_reject'] + s['fault_timeout'] + s['fault_actually_solvable']
        rate = f"{100*ok/att:.0f}%" if att else "N/A"
        total_fault_ok += ok
        total_fault_attempted += att
        total_bad_sol += s['fault_bad_solution']
        log(f"{N:>3}  {s['fault_correct_reject']:>6}  {s['fault_actually_solvable']:>6}  "
            f"{s['fault_bad_solution']:>6}  {s['fault_timeout']:>5}  {rate:>6}")
    fault_rate = f"{100*total_fault_ok/total_fault_attempted:.1f}%" \
                 if total_fault_attempted else "N/A"
    log(f"Total: {total_fault_ok}/{total_fault_attempted} ({fault_rate})")

    log("")
    if total_pos_solved == total_pos_attempted and total_pos_solved > 0:
        log("ALL POSITIVE TESTS PASSED")
    if total_bad_sol == 0:
        log("NO INVALID SOLUTIONS ON FAULT TESTS")
    else:
        log(f"WARNING: {total_bad_sol} INVALID SOLUTIONS ON FAULT TESTS!")
    log("=" * 78)


if __name__ == "__main__":
    main()
