from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from grading_harness import build_scale_suite
from hybrid_solver import hybrid_solve
from phase1.test_cases import verify_solution


CONFIG_PRESETS = {
    "best_known_5x5": {
        "timeout_nn": 12.0,
        "frontier_harvest_timeout": 8.0,
        "frontier_complete_timeout": 6.0,
        "frontier_complete_max_frontier_states": 4,
        "frontier_complete_max_nodes": 30000,
    },
    "deep_complete_5x5": {
        "timeout_nn": 12.0,
        "frontier_harvest_timeout": 8.0,
        "frontier_complete_timeout": 12.0,
        "frontier_complete_max_frontier_states": 6,
        "frontier_complete_max_nodes": 60000,
    },
    "no_harvest_deep_complete_5x5": {
        "timeout_nn": 12.0,
        "frontier_harvest_timeout": None,
        "frontier_complete_timeout": 12.0,
        "frontier_complete_max_frontier_states": 6,
        "frontier_complete_max_nodes": 60000,
        "frontier_harvest_search_profile": "__disable__",
    },
    "deep_harvest_complete_5x5": {
        "timeout_nn": 12.0,
        "frontier_harvest_timeout": 10.0,
        "frontier_complete_timeout": 12.0,
        "frontier_complete_max_frontier_states": 6,
        "frontier_complete_max_nodes": 60000,
    },
}


def run_config(model_name, cases, device, timeout_dlx, config_name):
    cfg = CONFIG_PRESETS[config_name]
    rows = []
    for case in cases:
        t0 = time.time()
        result = hybrid_solve(
            case.pieces,
            grid_size=case.grid_size,
            model_name=model_name,
            timeout_nn=cfg["timeout_nn"],
            timeout_dlx=timeout_dlx,
            device=device,
            verbose=False,
            frontier_harvest_timeout=cfg["frontier_harvest_timeout"],
            frontier_harvest_search_profile=cfg.get("frontier_harvest_search_profile"),
            frontier_complete_timeout=cfg["frontier_complete_timeout"],
            frontier_complete_max_frontier_states=cfg["frontier_complete_max_frontier_states"],
            frontier_complete_max_nodes=cfg["frontier_complete_max_nodes"],
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
    return {
        "config_name": config_name,
        "config": cfg,
        "valid_solved": valid_solved,
        "solve_rate": valid_solved / max(1, len(rows)),
        "avg_time_sec": sum(row["time_sec"] for row in rows) / max(1, len(rows)),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare structural 5x5 configs on the same case subset.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--eval-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1403)
    parser.add_argument("--timeout-dlx", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["best_known_5x5", "deep_complete_5x5", "deep_harvest_complete_5x5"],
        choices=sorted(CONFIG_PRESETS),
    )
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    cases, suite_source = build_scale_suite(
        grid_size=args.grid_size,
        n_cases=args.eval_cases,
        seed=args.seed,
        dlx_timeout=12.0,
    )
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [case for case in cases if case.case_id in wanted]

    report = {
        "model_name": args.model_name,
        "grid_size": args.grid_size,
        "suite_source": suite_source,
        "case_ids": [case.case_id for case in cases],
        "configs": [],
    }

    print("=" * 80)
    print("5x5 STRUCTURAL CONFIG COMPARISON")
    print("=" * 80)
    print(f"model={args.model_name} grid={args.grid_size} cases={[case.case_id for case in cases]}")

    for config_name in args.configs:
        result = run_config(
            model_name=args.model_name,
            cases=cases,
            device=args.device,
            timeout_dlx=args.timeout_dlx,
            config_name=config_name,
        )
        report["configs"].append(result)
        print(
            f"{config_name}: solve_rate={result['solve_rate']:.3f} "
            f"valid={result['valid_solved']}/{len(result['rows'])} "
            f"avg_time={result['avg_time_sec']:.3f}s"
        )
        for row in result["rows"]:
            print(
                f"  case={row['case_id']} solved={row['solved']} valid={row['valid']} "
                f"submethod={row['submethod']} time={row['time_sec']:.3f}s"
            )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
