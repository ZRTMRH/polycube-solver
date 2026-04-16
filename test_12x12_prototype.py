"""Benchmark slab planner v2 against the current 5x5x5 blockwise planner.

Loads the 40 12^3 solvable cases from ``fixtures/a_plus_150.json`` and runs:

  - current ``_solve_blockwise_5cube`` (which short-circuits at 12%5!=0, so
    effectively returns instantly with no solution)
  - new ``solve_slab_planner`` from ``block_planner_v2``

Reports counts, mean times, and per-case overlap.

Usage:
    python3 test_12x12_prototype.py            # full 40
    python3 test_12x12_prototype.py --subsample 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hybrid_solver import _solve_blockwise_5cube, _solve_line_rod_planner
from block_planner_v2 import solve_slab_planner


FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "a_plus_150.json"
)


def load_12_solvable(path: str = FIXTURE_PATH):
    data = json.load(open(path))
    cases = data["cases"]
    return [
        c for c in cases
        if c.get("grid_size") == 12 and c.get("expected_solvable")
    ]


def verify_solution(solution, pieces, grid_size):
    """Lightweight check: cells cover the full cube, no overlaps."""
    if solution is None:
        return False
    if len(solution) != len(pieces):
        return False
    used = set()
    for pidx, cells in solution.items():
        for cell in cells:
            if cell in used:
                return False
            x, y, z = cell
            if not (0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size):
                return False
            used.add(cell)
    return len(used) == grid_size ** 3


def run_current(case):
    """Mimic ``solve_size_gated``'s 12^3 large-planner path:
    rod-line planner first, then 5x5x5 blockwise (which rejects 12)."""
    pieces = case["pieces"]
    t0 = time.time()
    rod_sol = _solve_line_rod_planner(pieces, 12)
    if rod_sol is not None:
        return verify_solution(rod_sol, pieces, 12), time.time() - t0, {"reason": "rod_line"}

    sol, diag = _solve_blockwise_5cube(
        pieces=pieces,
        grid_size=12,
        device="cpu",
        block_timeout_dlx=8.0,
        block_timeout_nn=0.0,
        trials=3,
        retries_per_block=3,
    )
    elapsed = time.time() - t0
    ok = verify_solution(sol, pieces, 12)
    return ok, elapsed, diag


def run_v2(case):
    pieces = case["pieces"]
    t0 = time.time()
    sol, diag = solve_slab_planner(pieces, grid_size=12, slab_timeout=5.0)
    elapsed = time.time() - t0
    ok = verify_solution(sol, pieces, 12)
    return ok, elapsed, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=0,
                    help="run only first N cases (0 = all)")
    args = ap.parse_args()

    cases = load_12_solvable()
    if args.subsample and args.subsample < len(cases):
        cases = cases[: args.subsample]

    print(f"Loaded {len(cases)} 12^3 solvable cases")

    rows = []
    for c in cases:
        cur_ok, cur_t, cur_diag = run_current(c)
        v2_ok, v2_t, v2_diag = run_v2(c)
        rows.append({
            "case_id": c["case_id"],
            "generator": c["generator"],
            "current_ok": cur_ok,
            "current_t": cur_t,
            "current_reason": cur_diag.get("reason") if cur_diag else None,
            "v2_ok": v2_ok,
            "v2_t": v2_t,
            "v2_reason": v2_diag.get("reason") if v2_diag else None,
        })
        flag = lambda b: "OK" if b else "--"
        print(f"  {c['case_id']:<14} ({c['generator']:<22})  cur={flag(cur_ok)} ({cur_t:.2f}s) "
              f"v2={flag(v2_ok)} ({v2_t:.2f}s)  v2_reason={rows[-1]['v2_reason']}")

    n = len(rows)
    cur_n = sum(1 for r in rows if r["current_ok"])
    v2_n = sum(1 for r in rows if r["v2_ok"])
    cur_avg = sum(r["current_t"] for r in rows) / n
    v2_avg = sum(r["v2_t"] for r in rows) / n

    print()
    print("=" * 72)
    print(f"current: {cur_n}/{n}  avg time {cur_avg:.2f}s")
    print(f"v2     : {v2_n}/{n}  avg time {v2_avg:.2f}s")

    only_v2 = [r["case_id"] for r in rows if r["v2_ok"] and not r["current_ok"]]
    only_cur = [r["case_id"] for r in rows if r["current_ok"] and not r["v2_ok"]]
    print(f"solved by v2 only ({len(only_v2)}): {only_v2[:10]}{'...' if len(only_v2) > 10 else ''}")
    print(f"solved by current only ({len(only_cur)}): {only_cur[:10]}{'...' if len(only_cur) > 10 else ''}")

    fails_v2 = [(r["case_id"], r["v2_reason"]) for r in rows if not r["v2_ok"]]
    if fails_v2:
        print("\nv2 failures:")
        for cid, reason in fails_v2[:20]:
            print(f"  {cid}: {reason}")


if __name__ == "__main__":
    main()
