"""Build a representative 6x6 constructive benchmark suite.

The suite is intentionally split into three buckets:
1. `main_connected`: the primary headline metric
2. `legacy_style_hard`: cases selected to mimic the harsher old 5x5
   `dlxgen_*` distribution (chunkier, more repeated, fewer pieces)
3. `new_hard`: the hardest cases from the newer constructive families

This keeps a practical 6x6 benchmark grounded in what made the old cases
hard without pretending full old-style exact verification is a routine path
at 6x6 scale.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from phase1.polycube import ROTATIONS, get_all_placements, normalize, rotate
from phase2.data_generator import generate_constructive_puzzle_instances
from robust_generator import build_robust_constructive_case


def _json_default(obj):
    """Handle numpy-ish scalars without taking a hard dependency on numpy here."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _normalize_rotate_shuffle(pieces, seed):
    """Strip position/orientation/order bias from a case."""
    rng = random.Random(seed)
    out = [
        list(normalize(rotate(piece, rng.choice(ROTATIONS))))
        for piece in pieces
    ]
    rng.shuffle(out)
    return out


def _canonical_piece(piece):
    return tuple(sorted(tuple(cell) for cell in normalize(piece)))


def _canonical_case_key(grid_size, pieces):
    return (
        int(grid_size),
        tuple(sorted(_canonical_piece(piece) for piece in pieces)),
    )


def _difficulty_metrics(grid_size, pieces):
    sizes = [len(piece) for piece in pieces]
    placements = [len(get_all_placements(piece, grid_size)) for piece in pieces]
    shape_counts = Counter(_canonical_piece(piece) for piece in pieces)
    repeated_shape_fraction = (
        sum(v for v in shape_counts.values() if v > 1) / max(1, len(pieces))
    )

    metrics = {
        "num_pieces": len(pieces),
        "avg_piece_size": sum(sizes) / max(1, len(sizes)),
        "avg_piece_placements": sum(placements) / max(1, len(placements)),
        "max_piece_placements": max(placements) if placements else 0,
        "repeated_shape_fraction": repeated_shape_fraction,
        "unique_shapes": len(shape_counts),
    }
    # Legacy-style score leans toward the chunkier, lower-piece-count,
    # more-repeated shape bags that made the old dlxgen-style cases harsh.
    metrics["legacy_style_score"] = (
        3.4 * metrics["avg_piece_size"]
        + 4.0 * metrics["repeated_shape_fraction"]
        - 0.08 * metrics["num_pieces"]
        + 0.0004 * metrics["avg_piece_placements"]
    )
    # New-hard score emphasizes overall placement ambiguity a bit more.
    metrics["new_hard_score"] = (
        2.5 * metrics["avg_piece_size"]
        + 3.0 * metrics["repeated_shape_fraction"]
        - 0.05 * metrics["num_pieces"]
        + 0.0010 * metrics["avg_piece_placements"]
    )
    return metrics


def _build_robust_rotated_cases(grid_size, n_cases, seed):
    rows = []
    seen = set()
    attempts = 0
    max_attempts = max(100, n_cases * 30)

    while len(rows) < n_cases and attempts < max_attempts:
        attempts += 1
        case_seed = seed + attempts - 1
        pieces_abs = build_robust_constructive_case(grid_size, seed=case_seed)
        pieces = _normalize_rotate_shuffle(pieces_abs, case_seed)
        key = _canonical_case_key(grid_size, pieces)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "case_id": f"robust_rotated_{grid_size}_{len(rows):03d}",
            "seed": case_seed,
            "bucket": "robust_rotated",
            "generator_family": "robust_rotated",
            "instance_source": "robust_rotated",
            "pieces": pieces,
            "metrics": _difficulty_metrics(grid_size, pieces),
        })

    if len(rows) < n_cases:
        raise RuntimeError(
            f"Only built {len(rows)}/{n_cases} robust rotated cases for grid {grid_size}."
        )
    return rows


def _build_connected_cases(grid_size, n_cases, seed):
    instances = generate_constructive_puzzle_instances(
        num_instances=n_cases,
        grid_size=grid_size,
        seed=seed,
        large_suite_type="connected",
        verbose=False,
        allow_duplicate_fallback=True,
    )
    rows = []
    for idx, inst in enumerate(instances):
        case_seed = seed + idx
        pieces = _normalize_rotate_shuffle(inst["pieces"], case_seed)
        rows.append({
            "case_id": f"connected_{grid_size}_{idx:03d}",
            "seed": case_seed,
            "bucket": "main_connected",
            "generator_family": "constructive:connected",
            "instance_source": inst.get("instance_source"),
            "pieces": pieces,
            "metrics": _difficulty_metrics(grid_size, pieces),
        })
    return rows


def _build_hard_candidate_pool(grid_size, n_cases, seed, oversample_factor=6):
    """Build a pooled candidate set for hard-bucket selection."""
    target_pool = max(n_cases * oversample_factor, n_cases * 2)
    candidates = []
    seen = set()

    connected = generate_constructive_puzzle_instances(
        num_instances=target_pool,
        grid_size=grid_size,
        seed=seed,
        large_suite_type="connected",
        verbose=False,
        allow_duplicate_fallback=True,
    )
    for idx, inst in enumerate(connected):
        case_seed = seed + idx
        pieces = _normalize_rotate_shuffle(inst["pieces"], case_seed)
        key = _canonical_case_key(grid_size, pieces)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "seed": case_seed,
            "bucket": "hard_candidate_pool",
            "generator_family": "constructive:connected",
            "instance_source": inst.get("instance_source"),
            "pieces": pieces,
            "metrics": _difficulty_metrics(grid_size, pieces),
        })

    for idx in range(target_pool):
        case_seed = seed + 100000 + idx
        pieces_abs = build_robust_constructive_case(grid_size, seed=case_seed)
        pieces = _normalize_rotate_shuffle(pieces_abs, case_seed)
        key = _canonical_case_key(grid_size, pieces)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "seed": case_seed,
            "bucket": "hard_candidate_pool",
            "generator_family": "robust_rotated",
            "instance_source": "robust_rotated",
            "pieces": pieces,
            "metrics": _difficulty_metrics(grid_size, pieces),
        })
    return candidates


def _select_scored_cases(candidates, score_key, n_cases, bucket_name, grid_size, used_keys):
    ordered = sorted(
        candidates,
        key=lambda row: (
            row["metrics"][score_key],
            row["metrics"]["avg_piece_size"],
            row["metrics"]["repeated_shape_fraction"],
            -row["metrics"]["num_pieces"],
            row["metrics"]["avg_piece_placements"],
        ),
        reverse=True,
    )

    picked = []
    for row in ordered:
        case_key = _canonical_case_key(grid_size, row["pieces"])
        if case_key in used_keys:
            continue
        used_keys.add(case_key)
        picked.append({
            **row,
            "bucket": bucket_name,
            "case_id": f"{bucket_name}_{grid_size}_{len(picked):03d}",
        })
        if len(picked) >= n_cases:
            break

    if len(picked) < n_cases:
        raise RuntimeError(
            f"Only selected {len(picked)}/{n_cases} cases for bucket '{bucket_name}'."
        )
    return picked


def _bucket_summary(rows):
    if not rows:
        return {}
    metric_names = [
        "num_pieces",
        "avg_piece_size",
        "avg_piece_placements",
        "max_piece_placements",
        "repeated_shape_fraction",
        "unique_shapes",
        "legacy_style_score",
        "new_hard_score",
    ]
    summary = {"cases": len(rows)}
    for name in metric_names:
        summary[f"avg_{name}"] = round(
            statistics.mean(row["metrics"][name] for row in rows), 3
        )
    summary["sources"] = dict(Counter(row["instance_source"] for row in rows))
    summary["families"] = dict(Counter(row["generator_family"] for row in rows))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build a representative 6x6 constructive benchmark suite."
    )
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--connected-cases", type=int, default=20)
    parser.add_argument("--legacy-cases", type=int, default=10)
    parser.add_argument("--new-hard-cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="reports/6x6_benchmark_suite.json")
    args = parser.parse_args()

    connected_rows = _build_connected_cases(
        grid_size=args.grid_size,
        n_cases=args.connected_cases,
        seed=args.seed,
    )
    candidate_pool = _build_hard_candidate_pool(
        grid_size=args.grid_size,
        n_cases=max(args.legacy_cases, args.new_hard_cases),
        seed=args.seed + 10000,
    )
    connected_candidates = [
        row for row in candidate_pool
        if row["generator_family"] == "constructive:connected"
    ]
    used_keys = set()
    legacy_rows = _select_scored_cases(
        candidate_pool,
        score_key="legacy_style_score",
        n_cases=args.legacy_cases,
        bucket_name="legacy_style_hard",
        grid_size=args.grid_size,
        used_keys=used_keys,
    )
    new_hard_rows = _select_scored_cases(
        connected_candidates,
        score_key="new_hard_score",
        n_cases=args.new_hard_cases,
        bucket_name="new_hard",
        grid_size=args.grid_size,
        used_keys=used_keys,
    )

    all_rows = connected_rows + legacy_rows + new_hard_rows
    report = {
        "grid_size": args.grid_size,
        "suite_name": "representative_constructive_6x6",
        "design_notes": {
            "main_metric": "main_connected",
            "legacy_hard_metric": "legacy_style_hard",
            "new_hard_metric": "new_hard",
            "excluded_from_headline": ["constructive:striped", "constructive:mixed"],
        },
        "bucket_summaries": {
            "main_connected": _bucket_summary(connected_rows),
            "legacy_style_hard": _bucket_summary(legacy_rows),
            "new_hard": _bucket_summary(new_hard_rows),
        },
        "cases": all_rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=_json_default))

    print("=" * 80)
    print("6x6 BENCHMARK SUITE")
    print("=" * 80)
    for bucket_name in ("main_connected", "legacy_style_hard", "new_hard"):
        summary = report["bucket_summaries"][bucket_name]
        print(
            f"{bucket_name}: cases={summary['cases']} "
            f"avg_num_pieces={summary['avg_num_pieces']:.2f} "
            f"avg_piece_size={summary['avg_avg_piece_size']:.2f} "
            f"avg_repeated_shape_fraction={summary['avg_repeated_shape_fraction']:.3f} "
            f"avg_legacy_style_score={summary['avg_legacy_style_score']:.3f} "
            f"avg_new_hard_score={summary['avg_new_hard_score']:.3f}"
        )
    print(f"Saved suite to {out_path}")


if __name__ == "__main__":
    main()
