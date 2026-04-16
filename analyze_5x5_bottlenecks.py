"""
Step-by-step bottleneck analysis for larger-cube NN solving.

This script does not just report final solve rate. It walks through the actual
pipeline stages so we can distinguish between:
1. early branching explosion,
2. bad frontier quality (beam reaches states that are not recoverable), and
3. insufficient endgame search from a good frontier.

Example:
    .venv/bin/python -u analyze_5x5_bottlenecks.py \
        --model-name 5x5x5_modal_constructive_v1 \
        --grid-size 5 \
        --eval-cases 3 \
        --out reports/5x5_bottleneck_analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grading_harness import build_scale_suite
from phase2.nn_solver import (
    nn_solve,
    dfs_solve_from_frontier,
    complete_solve_from_frontier,
)
from phase2.search_profiles import (
    resolve_runtime_search_settings,
    resolve_retry_search_settings,
    resolve_structural_fallback_settings,
)
from phase2.train import load_model


def _stage_summary_nn(diag, num_pieces):
    depth = diag.get("depth_reached")
    return {
        "failure_reason": diag.get("failure_reason"),
        "depth_reached": depth,
        "pieces_total": num_pieces,
        "remaining_lower_bound": (num_pieces - depth) if depth is not None else None,
        "generated_children": diag.get("generated_children"),
        "placements_dropped_total": diag.get("placements_dropped_total"),
        "pocket_pruned_children": diag.get("pocket_pruned_children"),
        "expanded_states": diag.get("expanded_states"),
    }


def _infer_bottleneck(case_row):
    beam = case_row["beam"]
    retry = case_row.get("retry")
    dfs = case_row.get("dfs_frontier")
    complete = case_row.get("complete_frontier")
    wide_probe = case_row.get("wide_probe")

    # Fact-based classification, not a probabilistic guess.
    if complete and complete.get("solved"):
        return "recoverable_frontier_endgame_search"

    if complete:
        complete_reason = complete["diagnostics"].get("failure_reason")
        complete_nodes = complete["diagnostics"].get("nodes_expanded", 0) or 0
        if wide_probe and wide_probe.get("complete_frontier", {}).get("solved"):
            return "frontier_diversity_or_truncation_bottleneck"
        if complete_reason == "search_exhausted" and complete_nodes < 1000:
            return "top_frontier_states_locally_exhausted"
        if complete_reason in ("max_nodes", "timeout"):
            return "frontier_reachable_but_complete_budget_insufficient"

    beam_reason = beam["diagnostics"].get("failure_reason")
    beam_depth = beam["diagnostics"].get("depth_reached", 0)
    pieces_total = beam.get("pieces_total", 0)
    dropped = beam["diagnostics"].get("placements_dropped_total", 0) or 0

    if beam_reason == "timeout" and beam_depth <= max(2, pieces_total // 4):
        return "early_branching_explosion"

    if retry and retry["diagnostics"].get("failure_reason") == "timeout":
        if retry["diagnostics"].get("depth_reached", 0) <= beam_depth:
            return "retry_search_budget_not_helping"

    if beam_reason == "no_candidates" or (retry and retry["diagnostics"].get("failure_reason") == "no_candidates"):
        return "frontier_poisoned_by_search_policy"

    if dropped > 10000:
        return "placement_ranking_and_truncation_pressure"

    if dfs and dfs["diagnostics"].get("nodes_expanded", 0) > 0:
        return "beam_frontier_not_recoverable_under_stronger_local_search"

    return "undetermined_from_current_budget"


def analyze_case(case, model, max_pieces, device, timeout_nn, complete_timeout, complete_max_nodes,
                 probe_wide_frontier):
    grid_size = case.grid_size
    pieces = case.pieces
    num_pieces = len(pieces)

    runtime = resolve_runtime_search_settings(
        grid_size,
        default_beam_width=64,
        default_max_candidates_per_state=50,
    )
    retry_profile = resolve_retry_search_settings(grid_size)
    structural = resolve_structural_fallback_settings(grid_size) or {}

    beam_frontier = []
    beam_solution, beam_diag = nn_solve(
        pieces,
        grid_size,
        model,
        max_pieces=max_pieces,
        beam_width=runtime["beam_width"],
        timeout=timeout_nn,
        device=device,
        max_candidates_per_state=runtime["max_candidates_per_state"],
        enable_pocket_pruning=runtime["enable_pocket_pruning"],
        placement_ranker=runtime["placement_ranker"],
        max_children_per_layer=runtime["max_children_per_layer"],
        beam_diversity_slots=runtime["beam_diversity_slots"],
        beam_diversity_metric=runtime["beam_diversity_metric"],
        piece_branching_width=runtime["piece_branching_width"],
        piece_branching_slack=runtime["piece_branching_slack"],
        return_diagnostics=True,
        frontier_out=beam_frontier,
    )

    row = {
        "case_id": case.case_id,
        "pieces_total": num_pieces,
        "beam": {
            "profile": runtime,
            "solved": beam_solution is not None,
            "diagnostics": beam_diag,
            "summary": _stage_summary_nn(beam_diag, num_pieces),
            "frontier_size": len(beam_frontier),
        },
    }

    retry_frontier = []
    if retry_profile is not None:
        retry_solution, retry_diag = nn_solve(
            pieces,
            grid_size,
            model,
            max_pieces=max_pieces,
            beam_width=retry_profile["beam_width"],
            timeout=timeout_nn,
            device=device,
            max_candidates_per_state=retry_profile["max_candidates_per_state"],
            enable_pocket_pruning=retry_profile["enable_pocket_pruning"],
            placement_ranker=retry_profile["placement_ranker"],
            max_children_per_layer=retry_profile["max_children_per_layer"],
            beam_diversity_slots=retry_profile["beam_diversity_slots"],
            beam_diversity_metric=retry_profile["beam_diversity_metric"],
            piece_branching_width=retry_profile["piece_branching_width"],
            piece_branching_slack=retry_profile["piece_branching_slack"],
            return_diagnostics=True,
            frontier_out=retry_frontier,
        )
        row["retry"] = {
            "profile": retry_profile,
            "solved": retry_solution is not None,
            "diagnostics": retry_diag,
            "summary": _stage_summary_nn(retry_diag, num_pieces),
            "frontier_size": len(retry_frontier),
        }

    frontier_for_local = retry_frontier if retry_frontier else beam_frontier

    if frontier_for_local:
        dfs_solution, dfs_diag = dfs_solve_from_frontier(
            pieces=pieces,
            grid_size=grid_size,
            model=model,
            frontier_states=frontier_for_local,
            max_pieces=max_pieces,
            timeout=structural.get("dfs_timeout", 6.0),
            max_frontier_states=structural.get("dfs_max_frontier_states", 8),
            branch_limit=structural.get("dfs_branch_limit", 6),
            max_nodes=structural.get("dfs_max_nodes", 25000),
            device=device,
            placement_ranker=structural.get("dfs_placement_ranker", "contact"),
            enable_pocket_pruning=structural.get("dfs_enable_pocket_pruning", True),
            use_transposition=True,
        )
        row["dfs_frontier"] = {
            "solved": dfs_solution is not None,
            "diagnostics": dfs_diag,
        }

        complete_solution, complete_diag = complete_solve_from_frontier(
            pieces=pieces,
            grid_size=grid_size,
            model=model,
            frontier_states=frontier_for_local,
            max_pieces=max_pieces,
            timeout=complete_timeout,
            max_nodes=complete_max_nodes,
            max_frontier_states=4,
            device=device,
            placement_ranker="contact",
            enable_pocket_pruning=True,
            use_transposition=True,
        )
        row["complete_frontier"] = {
            "solved": complete_solution is not None,
            "diagnostics": complete_diag,
        }

    if probe_wide_frontier:
        wide_frontier = []
        wide_solution, wide_diag = nn_solve(
            pieces,
            grid_size,
            model,
            max_pieces=max_pieces,
            beam_width=12,
            timeout=timeout_nn,
            device=device,
            max_candidates_per_state=16,
            enable_pocket_pruning=True,
            placement_ranker="contact",
            max_children_per_layer=120,
            beam_diversity_slots=4,
            beam_diversity_metric="slice_profile",
            piece_branching_width=1,
            piece_branching_slack=0,
            return_diagnostics=True,
            frontier_out=wide_frontier,
        )
        wide_row = {
            "profile": {
                "beam_width": 12,
                "max_candidates_per_state": 16,
                "placement_ranker": "contact",
                "enable_pocket_pruning": True,
                "max_children_per_layer": 120,
                "beam_diversity_slots": 4,
                "beam_diversity_metric": "slice_profile",
                "piece_branching_width": 1,
                "piece_branching_slack": 0,
            },
            "solved": wide_solution is not None,
            "diagnostics": wide_diag,
            "summary": _stage_summary_nn(wide_diag, num_pieces),
            "frontier_size": len(wide_frontier),
        }
        if wide_frontier:
            wide_complete_solution, wide_complete_diag = complete_solve_from_frontier(
                pieces=pieces,
                grid_size=grid_size,
                model=model,
                frontier_states=wide_frontier,
                max_pieces=max_pieces,
                timeout=complete_timeout,
                max_nodes=complete_max_nodes,
                max_frontier_states=4,
                device=device,
                placement_ranker="contact",
                enable_pocket_pruning=True,
                use_transposition=True,
            )
            wide_row["complete_frontier"] = {
                "solved": wide_complete_solution is not None,
                "diagnostics": wide_complete_diag,
            }
        row["wide_probe"] = wide_row

    row["inferred_bottleneck"] = _infer_bottleneck(row)
    return row


def main():
    parser = argparse.ArgumentParser(description="Step-by-step bottleneck analysis for 5x5.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--eval-cases", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1403)
    parser.add_argument("--timeout-nn", type=float, default=8.0)
    parser.add_argument("--complete-timeout", type=float, default=20.0)
    parser.add_argument("--complete-max-nodes", type=int, default=100000)
    parser.add_argument("--no-wide-frontier-probe", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    model, _, metadata = load_model(args.model_name, device=args.device)
    max_pieces = model.in_channels - 1
    cases, suite_source = build_scale_suite(
        grid_size=args.grid_size,
        n_cases=args.eval_cases,
        seed=args.seed,
        dlx_timeout=12.0,
    )

    rows = [
        analyze_case(
            case=case,
            model=model,
            max_pieces=max_pieces,
            device=args.device,
            timeout_nn=args.timeout_nn,
            complete_timeout=args.complete_timeout,
            complete_max_nodes=args.complete_max_nodes,
            probe_wide_frontier=not args.no_wide_frontier_probe,
        )
        for case in cases
    ]

    bottleneck_counts = {}
    for row in rows:
        bottleneck_counts[row["inferred_bottleneck"]] = bottleneck_counts.get(row["inferred_bottleneck"], 0) + 1

    report = {
        "model_name": args.model_name,
        "grid_size": args.grid_size,
        "device": args.device,
        "timeout_nn": args.timeout_nn,
        "complete_timeout": args.complete_timeout,
        "complete_max_nodes": args.complete_max_nodes,
        "suite_source": suite_source,
        "eval_cases": len(cases),
        "model_metadata": metadata,
        "bottleneck_counts": bottleneck_counts,
        "rows": rows,
    }

    print("=" * 80)
    print("BOTTLENECK ANALYSIS")
    print("=" * 80)
    print(
        f"model={args.model_name} grid={args.grid_size} device={args.device} "
        f"suite_source={suite_source}"
    )
    print(f"cases={[case.case_id for case in cases]}")
    print(f"bottleneck_counts={bottleneck_counts}")
    for row in rows:
        beam = row["beam"]["summary"]
        print(
            f"case={row['case_id']} bottleneck={row['inferred_bottleneck']} "
            f"beam_reason={beam['failure_reason']} beam_depth={beam['depth_reached']}/"
            f"{beam['pieces_total']} dropped={beam['placements_dropped_total']}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
