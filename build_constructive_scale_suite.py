"""Build a constructive-only benchmark suite for larger cubes.

This intentionally ignores the legacy/DLX-style hard family and focuses on the
generator families that have already shown good scaling behavior:
1. `main_connected`: representative constructive performance
2. `robust_rotated`: orientation/order-bias reduced check
3. `new_hard`: hardest connected-derived constructive cases
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_6x6_benchmark_suite import (
    _bucket_summary,
    _build_connected_cases,
    _build_hard_candidate_pool,
    _build_robust_rotated_cases,
    _json_default,
    _select_scored_cases,
)


def build_suite(
    *,
    grid_size: int,
    connected_cases: int,
    robust_cases: int,
    new_hard_cases: int,
    seed: int,
):
    connected_rows = _build_connected_cases(
        grid_size=grid_size,
        n_cases=connected_cases,
        seed=seed,
    )
    robust_rows = _build_robust_rotated_cases(
        grid_size=grid_size,
        n_cases=robust_cases,
        seed=seed + 5000,
    )
    candidate_pool = _build_hard_candidate_pool(
        grid_size=grid_size,
        n_cases=new_hard_cases,
        seed=seed + 10000,
    )
    connected_candidates = [
        row for row in candidate_pool
        if row["generator_family"] == "constructive:connected"
    ]
    used_keys = set()
    new_hard_rows = _select_scored_cases(
        connected_candidates,
        score_key="new_hard_score",
        n_cases=new_hard_cases,
        bucket_name="new_hard",
        grid_size=grid_size,
        used_keys=used_keys,
    )

    return {
        "grid_size": grid_size,
        "suite_name": f"constructive_scale_{grid_size}x{grid_size}x{grid_size}",
        "design_notes": {
            "main_metric": "main_connected",
            "robust_metric": "robust_rotated",
            "new_hard_metric": "new_hard",
            "legacy_hard_excluded": True,
        },
        "bucket_summaries": {
            "main_connected": _bucket_summary(connected_rows),
            "robust_rotated": _bucket_summary(robust_rows),
            "new_hard": _bucket_summary(new_hard_rows),
        },
        "cases": connected_rows + robust_rows + new_hard_rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build a constructive-only benchmark suite for larger cubes."
    )
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--connected-cases", type=int, default=20)
    parser.add_argument("--robust-cases", type=int, default=10)
    parser.add_argument("--new-hard-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    report = build_suite(
        grid_size=args.grid_size,
        connected_cases=args.connected_cases,
        robust_cases=args.robust_cases,
        new_hard_cases=args.new_hard_cases,
        seed=args.seed,
    )

    out_path = Path(args.out) if args.out else Path(
        f"reports/{args.grid_size}x{args.grid_size}_constructive_scale_suite.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=_json_default))

    print("=" * 80)
    print("CONSTRUCTIVE SCALE SUITE")
    print("=" * 80)
    for bucket_name in ("main_connected", "robust_rotated", "new_hard"):
        summary = report["bucket_summaries"][bucket_name]
        print(
            f"{bucket_name}: cases={summary['cases']} "
            f"avg_num_pieces={summary['avg_num_pieces']:.2f} "
            f"avg_piece_size={summary['avg_avg_piece_size']:.2f} "
            f"avg_new_hard_score={summary['avg_new_hard_score']:.3f}"
        )
    print(f"Saved suite to {out_path}")


if __name__ == "__main__":
    main()
