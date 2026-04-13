"""
Benchmark hybrid_solver runtime behavior on constructive suites.

Example:
    .venv/bin/python -u benchmark_hybrid_profiles.py \
        --model-name 5x5x5_modal_constructive_v1 \
        --grid-size 5 \
        --eval-cases 3 \
        --timeout-nn 8 \
        --timeout-dlx 0.1 \
        --out reports/5x5_hybrid_profiles.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from grading_harness import build_scale_suite
from hybrid_solver import hybrid_solve
from phase1.test_cases import verify_solution


def main():
    parser = argparse.ArgumentParser(description="Benchmark hybrid solver profiles.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--eval-cases", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1403)
    parser.add_argument("--timeout-nn", type=float, default=8.0)
    parser.add_argument("--timeout-dlx", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--frontier-harvest-timeout", type=float, default=None)
    parser.add_argument("--frontier-complete-timeout", type=float, default=None)
    parser.add_argument("--frontier-complete-max-nodes", type=int, default=None)
    parser.add_argument("--frontier-complete-max-frontier-states", type=int, default=None)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    cases, suite_source = build_scale_suite(
        grid_size=args.grid_size,
        n_cases=args.eval_cases,
        seed=args.seed,
        dlx_timeout=12.0,
    )

    rows = []
    for case in cases:
        t0 = time.time()
        result = hybrid_solve(
            case.pieces,
            grid_size=case.grid_size,
            model_name=args.model_name,
            timeout_nn=args.timeout_nn,
            timeout_dlx=args.timeout_dlx,
            device=args.device,
            verbose=False,
            frontier_harvest_timeout=args.frontier_harvest_timeout,
            frontier_complete_timeout=args.frontier_complete_timeout,
            frontier_complete_max_nodes=args.frontier_complete_max_nodes,
            frontier_complete_max_frontier_states=args.frontier_complete_max_frontier_states,
        )
        elapsed = time.time() - t0
        solution = result.get("solution")
        solved = solution is not None
        valid = verify_solution(solution, case.grid_size) if solved else False
        rows.append({
            "case_id": case.case_id,
            "solved": solved,
            "valid": valid,
            "time_sec": elapsed,
            "method": result.get("method"),
            "submethod": result.get("submethod"),
            "stages_attempted": result.get("stages_attempted"),
        })

    valid_solved = sum(1 for row in rows if row["valid"])
    report = {
        "model_name": args.model_name,
        "grid_size": args.grid_size,
        "device": args.device,
        "timeout_nn": args.timeout_nn,
        "timeout_dlx": args.timeout_dlx,
        "suite_source": suite_source,
        "eval_cases": len(cases),
        "valid_solved": valid_solved,
        "solve_rate": valid_solved / max(1, len(rows)),
        "avg_time_sec": sum(row["time_sec"] for row in rows) / max(1, len(rows)),
        "rows": rows,
    }

    print("=" * 80)
    print("HYBRID PROFILE BENCHMARK")
    print("=" * 80)
    print(
        f"model={args.model_name} grid={args.grid_size} device={args.device} "
        f"timeout_nn={args.timeout_nn}"
    )
    print(f"suite_source={suite_source} cases={[case.case_id for case in cases]}")
    print(
        f"solve_rate={report['solve_rate']:.3f} valid={valid_solved}/{len(rows)} "
        f"avg_time={report['avg_time_sec']:.3f}s"
    )
    for row in rows:
        print(
            f"  case={row['case_id']} solved={row['solved']} valid={row['valid']} "
            f"time={row['time_sec']:.3f}s method={row['method']} submethod={row['submethod']} "
            f"stages={row['stages_attempted']}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
