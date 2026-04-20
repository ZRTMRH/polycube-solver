"""Benchmark a model against a fixed JSON suite definition.

Example:
    .venv/bin/python -u benchmark_fixed_suite.py \
        --model-name 6x6x6_frontier_adi_v1 \
        --suite reports/6x6_benchmark_suite_v2.json \
        --timeout-nn 24 \
        --timeout-dlx 0.1 \
        --out reports/6x6_frontier_adi_v1_eval.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from hybrid_solver import hybrid_solve
from phase1.test_cases import verify_solution


def _load_suite(path: str):
    suite_path = Path(path)
    data = json.loads(suite_path.read_text())
    if "cases" not in data:
        raise ValueError(f"Suite at {suite_path} has no 'cases' field.")
    return suite_path, data


def _evaluate_case(model_name, case, grid_size, timeout_nn, timeout_dlx, device):
    t0 = time.time()
    result = hybrid_solve(
        case["pieces"],
        grid_size=grid_size,
        model_name=model_name,
        timeout_nn=timeout_nn,
        timeout_dlx=timeout_dlx,
        device=device,
        verbose=False,
    )
    elapsed = time.time() - t0
    solution = result.get("solution")
    solved = solution is not None
    valid = verify_solution(solution, grid_size, case["pieces"]) if solved else False
    return {
        "case_id": case["case_id"],
        "bucket": case.get("bucket"),
        "generator_family": case.get("generator_family"),
        "instance_source": case.get("instance_source"),
        "solved": solved,
        "valid": valid,
        "time_sec": elapsed,
        "method": result.get("method"),
        "submethod": result.get("submethod"),
        "stages_attempted": result.get("stages_attempted"),
    }


def _bucket_summary(rows):
    valid_solved = sum(1 for row in rows if row["valid"])
    return {
        "cases": len(rows),
        "valid_solved": valid_solved,
        "solve_rate": valid_solved / max(1, len(rows)),
        "avg_time_sec": sum(row["time_sec"] for row in rows) / max(1, len(rows)),
        "submethods": dict(Counter(row.get("submethod") for row in rows)),
        "families": dict(Counter(row.get("generator_family") for row in rows)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark a model on a fixed JSON suite.")
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--suite", required=True, type=str)
    parser.add_argument("--timeout-nn", type=float, default=24.0)
    parser.add_argument("--timeout-dlx", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    suite_path, suite = _load_suite(args.suite)
    grid_size = int(suite["grid_size"])
    rows = [
        _evaluate_case(
            model_name=args.model_name,
            case=case,
            grid_size=grid_size,
            timeout_nn=args.timeout_nn,
            timeout_dlx=args.timeout_dlx,
            device=args.device,
        )
        for case in suite["cases"]
    ]

    by_bucket = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket"]].append(row)

    bucket_summaries = {
        bucket: _bucket_summary(bucket_rows)
        for bucket, bucket_rows in by_bucket.items()
    }

    overall = _bucket_summary(rows)
    report = {
        "model_name": args.model_name,
        "suite_path": str(suite_path),
        "suite_name": suite.get("suite_name"),
        "grid_size": grid_size,
        "timeout_nn": args.timeout_nn,
        "timeout_dlx": args.timeout_dlx,
        "device": args.device,
        "overall": overall,
        "bucket_summaries": bucket_summaries,
        "rows": rows,
    }

    print("=" * 80)
    print("FIXED SUITE BENCHMARK")
    print("=" * 80)
    print(
        f"model={args.model_name} grid={grid_size} suite={suite.get('suite_name')} "
        f"timeout_nn={args.timeout_nn} timeout_dlx={args.timeout_dlx}"
    )
    print(
        f"overall solve_rate={overall['solve_rate']:.3f} "
        f"valid={overall['valid_solved']}/{overall['cases']} "
        f"avg_time={overall['avg_time_sec']:.3f}s"
    )
    for bucket, summary in bucket_summaries.items():
        print(
            f"  {bucket}: solve_rate={summary['solve_rate']:.3f} "
            f"valid={summary['valid_solved']}/{summary['cases']} "
            f"avg_time={summary['avg_time_sec']:.3f}s"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
