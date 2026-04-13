"""
Benchmark named NN search profiles on a constructive held-out suite.

Example:
    .venv/bin/python -u benchmark_4x4_profiles.py \
        --model-name 4x4x4_adi2 \
        --grid-size 4 \
        --eval-cases 3 \
        --profiles default_like narrow_4x4 balanced_4x4 \
        --timeout 5 \
        --out reports/4x4_profiles.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from grading_harness import build_scale_suite
from phase1.test_cases import verify_solution
from phase2.nn_solver import nn_solve
from phase2.search_profiles import SEARCH_PROFILES, resolve_search_profile
from phase2.train import load_model


def benchmark_profile(model, cases, max_pieces, device, timeout, profile_name):
    profile = resolve_search_profile(profile_name)
    rows = []

    for case in cases:
        t0 = time.time()
        solution, diagnostics = nn_solve(
            case.pieces,
            case.grid_size,
            model,
            max_pieces=max_pieces,
            beam_width=profile["beam_width"],
            timeout=timeout,
            device=device,
            max_candidates_per_state=profile["max_candidates_per_state"],
            enable_pocket_pruning=profile["enable_pocket_pruning"],
            placement_ranker=profile["placement_ranker"],
            max_children_per_layer=profile["max_children_per_layer"],
            beam_diversity_slots=profile["beam_diversity_slots"],
            beam_diversity_metric=profile["beam_diversity_metric"],
            piece_branching_width=profile["piece_branching_width"],
            piece_branching_slack=profile["piece_branching_slack"],
            return_diagnostics=True,
        )
        elapsed = time.time() - t0
        solved = solution is not None
        valid = verify_solution(solution, case.grid_size) if solved else False
        rows.append({
            "case_id": case.case_id,
            "solved": solved,
            "valid": valid,
            "time_sec": elapsed,
            "failure_reason": diagnostics.get("failure_reason"),
            "depth_reached": diagnostics.get("depth_reached"),
            "expanded_states": diagnostics.get("expanded_states"),
            "generated_children": diagnostics.get("generated_children"),
            "pocket_pruned_children": diagnostics.get("pocket_pruned_children"),
            "placements_dropped_total": diagnostics.get("placements_dropped_total"),
        })

    valid_solved = sum(1 for row in rows if row["valid"])
    avg_time = sum(row["time_sec"] for row in rows) / max(1, len(rows))
    return {
        "profile_name": profile_name,
        "profile": profile,
        "n_cases": len(rows),
        "valid_solved": valid_solved,
        "solve_rate": valid_solved / max(1, len(rows)),
        "avg_time_sec": avg_time,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark named 4x4 search profiles.")
    parser.add_argument("--model-name", type=str, default="4x4x4_adi2")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--eval-cases", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1403)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--timeout-dlx-data", type=float, default=12.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["default_like", "narrow_4x4", "balanced_4x4"],
        choices=sorted(SEARCH_PROFILES),
    )
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    model, _, metadata = load_model(args.model_name, device=args.device)
    max_pieces = model.in_channels - 1

    cases, suite_source = build_scale_suite(
        grid_size=args.grid_size,
        n_cases=args.eval_cases,
        seed=args.seed,
        dlx_timeout=args.timeout_dlx_data,
    )

    report = {
        "model_name": args.model_name,
        "grid_size": args.grid_size,
        "device": args.device,
        "timeout": args.timeout,
        "suite_source": suite_source,
        "eval_cases": len(cases),
        "model_metadata": metadata,
        "profiles": [],
    }

    print("=" * 80)
    print("SEARCH PROFILE BENCHMARK")
    print("=" * 80)
    print(f"model={args.model_name} grid={args.grid_size} device={args.device}")
    print(f"suite_source={suite_source} cases={[case.case_id for case in cases]}")

    for profile_name in args.profiles:
        profile_report = benchmark_profile(
            model=model,
            cases=cases,
            max_pieces=max_pieces,
            device=args.device,
            timeout=args.timeout,
            profile_name=profile_name,
        )
        report["profiles"].append(profile_report)

        print("-" * 80)
        print(
            f"profile={profile_name} solve_rate={profile_report['solve_rate']:.3f} "
            f"valid={profile_report['valid_solved']}/{profile_report['n_cases']} "
            f"avg_time={profile_report['avg_time_sec']:.3f}s"
        )
        for row in profile_report["rows"]:
            print(
                f"  case={row['case_id']} solved={row['solved']} valid={row['valid']} "
                f"time={row['time_sec']:.3f}s reason={row['failure_reason']} "
                f"depth={row['depth_reached']} generated={row['generated_children']} "
                f"dropped={row['placements_dropped_total']}"
            )

    print("=" * 80)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
