from __future__ import annotations

import argparse
import json
from pathlib import Path

from grading_harness import build_scale_suite
from phase2.nn_solver import nn_solve, complete_solve_from_frontier
from phase2.search_profiles import (
    resolve_runtime_search_settings,
    resolve_retry_search_settings,
    resolve_search_profile,
)
from phase2.train import load_model


def _frontier_key(state_dict):
    occupied = frozenset(tuple(cell) for cell in state_dict.get("occupied", ()))
    remaining = tuple(sorted(int(i) for i in state_dict.get("remaining_indices", ())))
    return occupied, remaining


def _collect_frontier(model, pieces, grid_size, max_pieces, device, timeout, profile):
    frontier = []
    _, diagnostics = nn_solve(
        pieces,
        grid_size,
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
        frontier_out=frontier,
    )
    frontier.sort(key=lambda s: float(s.get("score", 0.0)), reverse=True)
    return frontier, diagnostics


def _audit_frontier_source(
    *,
    source_name,
    frontier,
    pieces,
    grid_size,
    model,
    max_pieces,
    device,
    root_timeout,
    root_max_nodes,
    root_placement_ranker,
):
    rows = []
    seen = {}
    for rank, root in enumerate(frontier, start=1):
        key = _frontier_key(root)
        duplicate_of = seen.get(key)
        if duplicate_of is None:
            seen[key] = rank

        solved, diag = complete_solve_from_frontier(
            pieces=pieces,
            grid_size=grid_size,
            model=model,
            frontier_states=[root],
            max_pieces=max_pieces,
            timeout=root_timeout,
            max_nodes=root_max_nodes,
            max_frontier_states=1,
            device=device,
            placement_ranker=root_placement_ranker,
            enable_pocket_pruning=True,
            use_transposition=True,
        )
        rows.append(
            {
                "source": source_name,
                "rank": rank,
                "score": float(root.get("score", 0.0)),
                "remaining_count": len(root.get("remaining_indices", ())),
                "duplicate_of_rank": duplicate_of,
                "solvable_from_root": solved is not None,
                "complete_diag": diag,
            }
        )
    return rows


def _summarize_root_rows(rows):
    if not rows:
        return {
            "n_roots": 0,
            "n_solvable_roots": 0,
            "best_solvable_rank": None,
            "best_solvable_score": None,
            "top_root_solvable": False,
            "ranking_gap_over_best_solvable": None,
        }

    solvable = [row for row in rows if row["solvable_from_root"]]
    best_solvable = min(solvable, key=lambda row: row["rank"]) if solvable else None
    top = rows[0]
    gap = None
    if best_solvable is not None:
        gap = top["score"] - best_solvable["score"]
    return {
        "n_roots": len(rows),
        "n_solvable_roots": len(solvable),
        "best_solvable_rank": None if best_solvable is None else best_solvable["rank"],
        "best_solvable_score": None if best_solvable is None else best_solvable["score"],
        "top_root_solvable": bool(top["solvable_from_root"]),
        "ranking_gap_over_best_solvable": gap,
    }


def audit_case(
    *,
    case,
    model,
    max_pieces,
    device,
    timeout_nn,
    root_timeout,
    root_max_nodes,
    root_placement_ranker,
    frontier_profiles,
):
    rows = {
        "case_id": case.case_id,
        "sources": {},
    }

    all_roots = []
    for source_name, profile in frontier_profiles:
        if profile is None:
            continue
        frontier, diagnostics = _collect_frontier(
            model=model,
            pieces=case.pieces,
            grid_size=case.grid_size,
            max_pieces=max_pieces,
            device=device,
            timeout=timeout_nn if source_name != "harvest" else max(timeout_nn, 8.0),
            profile=profile,
        )
        source_rows = _audit_frontier_source(
            source_name=source_name,
            frontier=frontier,
            pieces=case.pieces,
            grid_size=case.grid_size,
            model=model,
            max_pieces=max_pieces,
            device=device,
            root_timeout=root_timeout,
            root_max_nodes=root_max_nodes,
            root_placement_ranker=root_placement_ranker,
        )
        rows["sources"][source_name] = {
            "profile": profile,
            "search_diag": diagnostics,
            "summary": _summarize_root_rows(source_rows),
            "roots": source_rows,
        }
        all_roots.extend(source_rows)

    global_solvable = [row for row in all_roots if row["solvable_from_root"]]
    global_best = min(global_solvable, key=lambda row: row["score"], default=None)
    rows["global_summary"] = {
        "n_sources": len(rows["sources"]),
        "n_total_roots": len(all_roots),
        "n_total_solvable_roots": len(global_solvable),
        "best_source_by_ranked_solvable": (
            min(
                (
                    (name, data["summary"]["best_solvable_rank"])
                    for name, data in rows["sources"].items()
                    if data["summary"]["best_solvable_rank"] is not None
                ),
                key=lambda item: item[1],
                default=(None, None),
            )[0]
        ),
        "highest_score_solvable_source": None if global_best is None else global_best["source"],
    }
    return rows


def main():
    parser = argparse.ArgumentParser(description="Audit retry/harvest frontier roots on 5x5 cases.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--eval-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1403)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--timeout-nn", type=float, default=12.0)
    parser.add_argument("--root-timeout", type=float, default=12.0)
    parser.add_argument("--root-max-nodes", type=int, default=60000)
    parser.add_argument("--root-placement-ranker", type=str, default="contact")
    parser.add_argument(
        "--profile-spec",
        nargs="*",
        default=None,
        help="Optional source=profile pairs to audit, e.g. beam=ultra_capped_5x5 branch=piece_branch_5x5",
    )
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
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [case for case in cases if case.case_id in wanted]

    if args.profile_spec:
        frontier_profiles = []
        for spec in args.profile_spec:
            source_name, profile_name = spec.split("=", 1)
            frontier_profiles.append((source_name, resolve_search_profile(profile_name)))
    else:
        frontier_profiles = [
            ("beam", resolve_runtime_search_settings(args.grid_size)),
            ("retry", resolve_retry_search_settings(args.grid_size)),
            ("harvest", resolve_search_profile("harvest_5x5")),
        ]

    rows = [
        audit_case(
            case=case,
            model=model,
            max_pieces=max_pieces,
            device=args.device,
            timeout_nn=args.timeout_nn,
            root_timeout=args.root_timeout,
            root_max_nodes=args.root_max_nodes,
            root_placement_ranker=args.root_placement_ranker,
            frontier_profiles=frontier_profiles,
        )
        for case in cases
    ]

    report = {
        "model_name": args.model_name,
        "grid_size": args.grid_size,
        "suite_source": suite_source,
        "case_ids": [case.case_id for case in cases],
        "frontier_profiles": [
            {"source": source_name, "profile": profile}
            for source_name, profile in frontier_profiles if profile is not None
        ],
        "model_metadata": metadata,
        "rows": rows,
    }

    print("=" * 80)
    print("5x5 FRONTIER ROOT AUDIT")
    print("=" * 80)
    print(f"model={args.model_name} cases={[case.case_id for case in cases]}")
    for row in rows:
        print(f"case={row['case_id']} total_solvable_roots={row['global_summary']['n_total_solvable_roots']}")
        for source_name, source in row["sources"].items():
            summary = source["summary"]
            print(
                f"  {source_name}: roots={summary['n_roots']} "
                f"solvable={summary['n_solvable_roots']} "
                f"best_rank={summary['best_solvable_rank']} "
                f"top_root_solvable={summary['top_root_solvable']}"
            )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
