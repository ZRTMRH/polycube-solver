#!/usr/bin/env python3
"""
Run our solve_size_gated() against partner's eval_4_to_12.json fixture.
Usage: ~/venv/bin/python3 run_partner_eval.py [--fixture PATH] [--budget SECONDS]
"""
import json, sys, time, signal, threading, os
from pathlib import Path
from collections import defaultdict

FIXTURE = Path("/tmp/eval_4_to_12.json")
BUDGET = 4 * 3600  # 4 hour total budget

# ── helpers ──────────────────────────────────────────────────────────────────

def solve_with_wall_timeout(pieces, grid_size, wall_timeout, solve_timeout, dlx_timeout):
    """Run solve_size_gated with a hard wall-clock timeout via SIGUSR1."""
    from hybrid_solver import solve_size_gated

    class WallTimeout(Exception):
        pass

    old_handler = signal.getsignal(signal.SIGUSR1)
    def _handler(signum, frame):
        raise WallTimeout()

    signal.signal(signal.SIGUSR1, _handler)
    timer = threading.Timer(wall_timeout, os.kill, args=(os.getpid(), signal.SIGUSR1))
    timer.start()
    try:
        result = solve_size_gated(
            pieces,
            grid_size=grid_size,
            model_name="auto",
            timeout=solve_timeout,
            timeout_dlx=dlx_timeout,
            verbose=False,
        )
        return result
    except (WallTimeout, BaseException) as e:
        return {"solution": None, "submethod": f"wall_timeout:{type(e).__name__}"}
    finally:
        timer.cancel()
        signal.signal(signal.SIGUSR1, old_handler)


def verify_solution(solution, grid_size, pieces):
    """Independent solution verification."""
    from phase1.polycube import normalize, get_orientations

    expected = {(x, y, z) for x in range(grid_size)
                for y in range(grid_size) for z in range(grid_size)}
    covered = set()

    for pidx, cells in solution.items():
        cell_set = set()
        for c in cells:
            t = tuple(c) if not isinstance(c, tuple) else c
            assert 0 <= t[0] < grid_size and 0 <= t[1] < grid_size and 0 <= t[2] < grid_size, \
                f"Out of bounds: {t}"
            cell_set.add(t)

        assert cell_set.isdisjoint(covered), f"Overlap in piece {pidx}"
        covered.update(cell_set)

        # Shape match
        placed_norm = normalize(frozenset(cell_set))
        orig = pieces[int(pidx)] if isinstance(pidx, str) else pieces[pidx]
        orig_fs = frozenset(tuple(c) for c in orig)
        orientations = get_orientations(orig_fs)
        assert placed_norm in orientations, f"Shape mismatch for piece {pidx}"

    assert covered == expected, f"Coverage: {len(covered)}/{len(expected)}"
    return True


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    from hybrid_solver import recommended_timeouts

    print(f"Loading fixture from {FIXTURE} ...")
    data = json.loads(FIXTURE.read_text())
    cases = data["cases"]
    print(f"Loaded {len(cases)} cases (grid sizes {data['grid_sizes']})")
    print(f"Solvable: {sum(1 for c in cases if c['expected_solvable'])}, "
          f"Unsolvable: {sum(1 for c in cases if not c['expected_solvable'])}")
    print(f"Budget: {BUDGET}s\n")

    results = []
    t_start = time.time()

    for i, case in enumerate(cases):
        elapsed_total = time.time() - t_start
        if elapsed_total > BUDGET:
            print(f"\n*** BUDGET EXHAUSTED after {i} cases ***")
            break

        cid = case["case_id"]
        gs = case["grid_size"]
        expected = case["expected_solvable"]
        pieces = [[tuple(c) for c in p] for p in case["pieces"]]

        gen_t, solve_t, dlx_t = recommended_timeouts(gs)
        wall_t = solve_t + 60  # extra margin

        t0 = time.time()
        raw = solve_with_wall_timeout(pieces, gs, wall_t, solve_t, dlx_t)
        dt = time.time() - t0

        solution = raw.get("solution") if isinstance(raw, dict) else None
        submethod = raw.get("submethod", "") if isinstance(raw, dict) else ""

        correct = False
        reason = ""
        if expected:
            if solution is None:
                reason = f"missed (via {submethod})"
            else:
                try:
                    verify_solution(solution, gs, pieces)
                    correct = True
                except (AssertionError, Exception) as e:
                    reason = f"invalid: {e}"
        else:
            correct = (solution is None)
            if not correct:
                reason = "solved unsolvable"

        results.append({
            "case_id": cid, "grid_size": gs, "expected": expected,
            "correct": correct, "time": dt, "submethod": submethod, "reason": reason,
        })

        status = "OK" if correct else f"FAIL ({reason})"
        tag = "solv" if expected else "unsolv"
        print(f"[{i+1:3d}/{len(cases)}] {cid:25s} N={gs:2d} {tag:6s} {dt:6.1f}s {submethod:30s} {status}")

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    print(f"OVERALL: {correct_count}/{total} ({correct_count/total:.1%})")

    solv = [r for r in results if r["expected"]]
    unsolv = [r for r in results if not r["expected"]]
    solv_ok = sum(1 for r in solv if r["correct"])
    unsolv_ok = sum(1 for r in unsolv if r["correct"])
    print(f"  Solvable:   {solv_ok}/{len(solv)} ({solv_ok/len(solv):.1%})" if solv else "")
    print(f"  Unsolvable: {unsolv_ok}/{len(unsolv)} ({unsolv_ok/len(unsolv):.1%})" if unsolv else "")

    print(f"\nBy grid size:")
    by_gs = defaultdict(list)
    for r in results:
        by_gs[r["grid_size"]].append(r)
    for gs in sorted(by_gs):
        rows = by_gs[gs]
        ok = sum(1 for r in rows if r["correct"])
        avg_t = sum(r["time"] for r in rows) / len(rows)
        s_rows = [r for r in rows if r["expected"]]
        u_rows = [r for r in rows if not r["expected"]]
        s_ok = sum(1 for r in s_rows if r["correct"])
        u_ok = sum(1 for r in u_rows if r["correct"])
        print(f"  N={gs:2d}: {ok}/{len(rows)} ({ok/len(rows):.0%})  "
              f"[solv {s_ok}/{len(s_rows)}, unsolv {u_ok}/{len(u_rows)}]  "
              f"avg {avg_t:.1f}s")

    # failures
    fails = [r for r in results if not r["correct"]]
    if fails:
        print(f"\nFailed cases ({len(fails)}):")
        for r in fails:
            print(f"  {r['case_id']} N={r['grid_size']} {r['reason']}")

    print(f"\nTotal wall time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
