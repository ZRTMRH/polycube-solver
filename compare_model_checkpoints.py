"""
Compare multiple model checkpoints on the same benchmark suite.

Example:
    .venv/bin/python -u compare_model_checkpoints.py \
        --models 4x4x4_modal_constructive 4x4x4_modal_constructive_v2 \
        --grid-size 4 \
        --mode nn \
        --eval-cases 10 \
        --timeout 5 \
        --out reports/compare_4x4_models.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from grading_harness import build_scale_suite
from hybrid_solver import hybrid_solve
from phase1.test_cases import verify_solution
from phase2.nn_solver import solve_with_nn


def evaluate_model(model_name, cases, mode, timeout, timeout_dlx, device):
    rows = []
    for case in cases:
        t0 = time.time()
        if mode == "hybrid":
            result = hybrid_solve(
                case.pieces,
                grid_size=case.grid_size,
                model_name=model_name,
                timeout_nn=timeout,
                timeout_dlx=timeout_dlx,
                device=device,
                verbose=False,
            )
            solution = result.get("solution")
            method = result.get("method")
            submethod = result.get("submethod")
            stages = result.get("stages_attempted")
        else:
            solution = solve_with_nn(
                case.pieces,
                grid_size=case.grid_size,
                model_name=model_name,
                timeout=timeout,
                device=device,
            )
            method = "nn" if solution is not None else None
            submethod = "beam" if solution is not None else "nn_failed"
            stages = None

        elapsed = time.time() - t0
        solved = solution is not None
        valid = verify_solution(solution, case.grid_size) if solved else False
        rows.append({
            "case_id": case.case_id,
            "solved": solved,
            "valid": valid,
            "time_sec": elapsed,
            "method": method,
            "submethod": submethod,
            "stages_attempted": stages,
        })

    valid_solved = sum(1 for row in rows if row["valid"])
    return {
        "model_name": model_name,
        "n_cases": len(rows),
        "valid_solved": valid_solved,
        "solve_rate": valid_solved / max(1, len(rows)),
        "avg_time_sec": sum(row["time_sec"] for row in rows) / max(1, len(rows)),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare multiple model checkpoints.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--mode", choices=["nn", "hybrid"], default="nn")
    parser.add_argument("--eval-cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1403)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--timeout-dlx", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    cases, suite_source = build_scale_suite(
        grid_size=args.grid_size,
        n_cases=args.eval_cases,
        seed=args.seed,
        dlx_timeout=12.0,
    )

    report = {
        "grid_size": args.grid_size,
        "mode": args.mode,
        "device": args.device,
        "timeout": args.timeout,
        "timeout_dlx": args.timeout_dlx,
        "suite_source": suite_source,
        "eval_cases": len(cases),
        "models": [],
    }

    print("=" * 80)
    print("MODEL CHECKPOINT COMPARISON")
    print("=" * 80)
    print(
        f"grid={args.grid_size} mode={args.mode} device={args.device} "
        f"timeout={args.timeout} suite_source={suite_source}"
    )
    print(f"cases={[case.case_id for case in cases]}")

    for model_name in args.models:
        result = evaluate_model(
            model_name=model_name,
            cases=cases,
            mode=args.mode,
            timeout=args.timeout,
            timeout_dlx=args.timeout_dlx,
            device=args.device,
        )
        report["models"].append(result)
        print(
            f"model={model_name} solve_rate={result['solve_rate']:.3f} "
            f"valid={result['valid_solved']}/{result['n_cases']} "
            f"avg_time={result['avg_time_sec']:.3f}s"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
