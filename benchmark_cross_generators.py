"""
Benchmark a model/search setup across multiple 5x5 generator families.

This is meant to answer a narrow but important question:
does a strong solve rate hold across several constructive generators, or is it
mostly a match to one particular case distribution?

Example:
    .venv/bin/python -u benchmark_cross_generators.py \
        --model-name 5x5x5_calibrated_v1_light \
        --grid-size 5 \
        --eval-cases 10 \
        --timeout-nn 24 \
        --timeout-dlx 0.1 \
        --out reports/5x5_cross_generators_t24.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from grading_harness import build_scale_suite
from hybrid_solver import hybrid_solve
from phase1.polycube import ROTATIONS, normalize, rotate
from phase1.test_cases import verify_solution
from phase2.data_generator import (
    enumerate_polycubes,
    generate_constructive_puzzle_instances,
    generate_puzzle_instances,
)
from robust_generator import build_robust_constructive_case


DEFAULT_CASE_SOURCES = [
    "dlx_random_verified",
    "greedy_random",
    "robust_rotated",
    "constructive:robust",
    "constructive:connected",
    "constructive:mixed",
    "constructive:striped",
]


def _checkpoint_path(model_name: str) -> Path:
    return Path("phase2/trained_models") / f"{model_name}.pt"


def _randomize_piece_orientations(pieces, seed):
    """Normalize each piece and apply a deterministic random rotation."""
    rng = random.Random(seed)
    return [
        list(normalize(rotate(piece, rng.choice(ROTATIONS))))
        for piece in pieces
    ]


def _build_cases(case_source: str, grid_size: int, eval_cases: int, seed: int):
    """Return [(case_id, case_seed, pieces), ...] for one generator family."""
    if case_source == "dlx_random_verified":
        catalog = enumerate_polycubes(max_size=5)
        instances = generate_puzzle_instances(
            num_instances=eval_cases,
            grid_size=grid_size,
            polycube_catalog=catalog,
            min_piece_size=3,
            max_piece_size=5,
            dlx_timeout=12.0,
            verbose=False,
            seed=seed,
        )
        return case_source, [
            {
                "case_id": f"dlx_verified_{idx:02d}",
                "case_seed": seed + idx,
                "pieces": inst["pieces"],
                "instance_source": "dlx_random_verified",
            }
            for idx, inst in enumerate(instances)
        ]

    if case_source == "greedy_random":
        cases, suite_source = build_scale_suite(
            grid_size=grid_size,
            n_cases=eval_cases,
            seed=seed,
            dlx_timeout=12.0,
        )
        return suite_source, [
            {
                "case_id": case.case_id,
                "case_seed": seed + idx,
                "pieces": case.pieces,
            }
            for idx, case in enumerate(cases)
        ]

    if case_source == "robust_rotated":
        rows = []
        for idx in range(eval_cases):
            case_seed = seed + idx
            pieces_abs = build_robust_constructive_case(grid_size, seed=case_seed)
            pieces = _randomize_piece_orientations(pieces_abs, case_seed)
            rows.append(
                {
                    "case_id": f"robust_rotated_{grid_size}_{case_seed}",
                    "case_seed": case_seed,
                    "pieces": pieces,
                }
            )
        return case_source, rows

    prefix = "constructive:"
    if case_source.startswith(prefix):
        variant = case_source[len(prefix):]
        instances = generate_constructive_puzzle_instances(
            num_instances=eval_cases,
            grid_size=grid_size,
            seed=seed,
            large_suite_type=variant,
            verbose=False,
            allow_duplicate_fallback=True,
        )
        rows = []
        for idx, inst in enumerate(instances):
            case_seed = seed + idx
            pieces = _randomize_piece_orientations(inst["pieces"], case_seed)
            rows.append(
                {
                    "case_id": f"{variant}_{idx:02d}",
                    "case_seed": case_seed,
                    "pieces": pieces,
                    "instance_source": inst.get("instance_source"),
                }
            )
        return case_source, rows

    raise ValueError(f"Unknown case source: {case_source}")


def _run_case(model_name, grid_size, timeout_nn, timeout_dlx, device, case):
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
        "seed": case["case_seed"],
        "num_pieces": len(case["pieces"]),
        "instance_source": case.get("instance_source"),
        "solved": solved,
        "valid": valid,
        "time_sec": elapsed,
        "method": result.get("method"),
        "submethod": result.get("submethod"),
        "stages_attempted": result.get("stages_attempted"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark one model/search setup across multiple generator families."
    )
    parser.add_argument("--model-name", type=str, default="5x5x5_calibrated_v1_light")
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--eval-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-nn", type=float, default=24.0)
    parser.add_argument("--timeout-dlx", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--case-sources",
        nargs="+",
        default=DEFAULT_CASE_SOURCES,
        help=(
            "Generator families to benchmark. "
            "Use names like greedy_random, robust_rotated, constructive:robust, "
            "constructive:connected, constructive:mixed, constructive:striped."
        ),
    )
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    checkpoint_path = _checkpoint_path(args.model_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Restore the checkpoint locally, then rerun."
        )

    generator_reports = []
    for source_idx, case_source in enumerate(args.case_sources):
        source_seed = args.seed + source_idx * 1000
        suite_name, cases = _build_cases(
            case_source=case_source,
            grid_size=args.grid_size,
            eval_cases=args.eval_cases,
            seed=source_seed,
        )
        rows = [
            _run_case(
                model_name=args.model_name,
                grid_size=args.grid_size,
                timeout_nn=args.timeout_nn,
                timeout_dlx=args.timeout_dlx,
                device=args.device,
                case=case,
            )
            for case in cases
        ]
        valid_solved = sum(1 for row in rows if row["valid"])
        avg_time_sec = sum(row["time_sec"] for row in rows) / max(1, len(rows))
        generator_reports.append(
            {
                "case_source": case_source,
                "suite_name": suite_name,
                "seed": source_seed,
                "eval_cases": len(rows),
                "valid_solved": valid_solved,
                "solve_rate": valid_solved / max(1, len(rows)),
                "avg_time_sec": avg_time_sec,
                "rows": rows,
            }
        )

    report = {
        "model_name": args.model_name,
        "grid_size": args.grid_size,
        "device": args.device,
        "timeout_nn": args.timeout_nn,
        "timeout_dlx": args.timeout_dlx,
        "eval_cases_per_source": args.eval_cases,
        "generator_reports": generator_reports,
    }

    print("=" * 80)
    print("CROSS GENERATOR BENCHMARK")
    print("=" * 80)
    print(
        f"model={args.model_name} grid={args.grid_size} device={args.device} "
        f"timeout_nn={args.timeout_nn} timeout_dlx={args.timeout_dlx}"
    )
    for item in generator_reports:
        print(
            f"  source={item['case_source']} suite={item['suite_name']} "
            f"valid={item['valid_solved']}/{item['eval_cases']} "
            f"solve_rate={item['solve_rate']:.3f} avg_time={item['avg_time_sec']:.3f}s"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
