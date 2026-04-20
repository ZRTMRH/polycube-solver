"""
Benchmark a trained model on robust-generated, randomly rotated cases.

This is meant to test the current best 5x5 baseline under the newer case
generation path:
  robust_generator -> normalize+random-rotate -> hybrid_solve -> verify

Example:
    python -u benchmark_robust_generated_cases.py \
        --model-name 5x5x5_calibrated_v1_light \
        --grid-size 5 \
        --eval-cases 10 \
        --timeout-nn 24 \
        --out reports/5x5_v1_light_robust_t24.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from hybrid_solver import hybrid_solve
from phase1.polycube import ROTATIONS, normalize, rotate
from phase1.test_cases import verify_solution
from robust_generator import build_robust_constructive_case


def _checkpoint_path(model_name: str) -> Path:
    return Path("phase2/trained_models") / f"{model_name}.pt"


def _build_rotated_case(grid_size: int, seed: int):
    """Generate one robust constructive case and strip position/orientation bias."""
    pieces_abs = build_robust_constructive_case(grid_size, seed=seed)
    rng = random.Random(seed)
    pieces = [
        list(normalize(rotate(piece, rng.choice(ROTATIONS))))
        for piece in pieces_abs
    ]
    return pieces


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark hybrid_solve on robust-generated rotated cases."
    )
    parser.add_argument("--model-name", type=str, default="5x5x5_calibrated_v1_light")
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--eval-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-nn", type=float, default=24.0)
    parser.add_argument("--timeout-dlx", type=float, default=120.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    checkpoint_path = _checkpoint_path(args.model_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Download or restore the checkpoint locally, then rerun."
        )

    rows = []
    generated = 0
    valid_solved = 0
    solve_times = []

    for idx in range(args.eval_cases):
        case_seed = args.seed + idx
        case_id = f"robust_rotated_{args.grid_size}_{case_seed}"

        try:
            pieces = _build_rotated_case(args.grid_size, case_seed)
            generated += 1
        except RuntimeError as exc:
            rows.append({
                "case_id": case_id,
                "seed": case_seed,
                "generated": False,
                "solved": False,
                "valid": False,
                "time_sec": 0.0,
                "num_pieces": None,
                "method": None,
                "submethod": "generator_error",
                "reason": str(exc),
            })
            continue

        t0 = time.time()
        result = hybrid_solve(
            pieces,
            grid_size=args.grid_size,
            model_name=args.model_name,
            timeout_nn=args.timeout_nn,
            timeout_dlx=args.timeout_dlx,
            device=args.device,
            verbose=False,
        )
        elapsed = time.time() - t0
        solve_times.append(elapsed)

        solution = result.get("solution")
        solved = solution is not None
        valid = verify_solution(solution, args.grid_size, pieces) if solved else False
        if valid:
            valid_solved += 1

        rows.append({
            "case_id": case_id,
            "seed": case_seed,
            "generated": True,
            "solved": solved,
            "valid": valid,
            "time_sec": elapsed,
            "num_pieces": len(pieces),
            "method": result.get("method"),
            "submethod": result.get("submethod"),
            "stages_attempted": result.get("stages_attempted"),
        })

    attempted = sum(1 for row in rows if row["generated"])
    report = {
        "model_name": args.model_name,
        "grid_size": args.grid_size,
        "device": args.device,
        "timeout_nn": args.timeout_nn,
        "timeout_dlx": args.timeout_dlx,
        "case_source": "robust_generator_rotated",
        "eval_cases_requested": args.eval_cases,
        "cases_generated": generated,
        "cases_attempted": attempted,
        "valid_solved": valid_solved,
        "solve_rate": valid_solved / max(1, attempted),
        "avg_time_sec": sum(solve_times) / max(1, len(solve_times)),
        "rows": rows,
    }

    print("=" * 80)
    print("ROBUST GENERATED CASE BENCHMARK")
    print("=" * 80)
    print(
        f"model={args.model_name} grid={args.grid_size} device={args.device} "
        f"timeout_nn={args.timeout_nn} timeout_dlx={args.timeout_dlx}"
    )
    print(
        f"case_source={report['case_source']} generated={generated}/{args.eval_cases} "
        f"valid={valid_solved}/{attempted} solve_rate={report['solve_rate']:.3f}"
    )
    for row in rows:
        if not row["generated"]:
            print(f"  case={row['case_id']} generated=False reason={row['reason']}")
            continue
        print(
            f"  case={row['case_id']} solved={row['solved']} valid={row['valid']} "
            f"time={row['time_sec']:.3f}s method={row['method']} "
            f"submethod={row['submethod']} stages={row['stages_attempted']}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
