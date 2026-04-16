"""
Audit diversity of constructive puzzle-instance generation.

Example:
    .venv/bin/python -u audit_constructive_diversity.py \
        --grid-size 5 \
        --num-instances 80 \
        --seed 42 \
        --out reports/constructive_diversity_5x5.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from phase2.data_generator import (
    _canonical_instance_key,
    generate_constructive_puzzle_instances,
)


def _piece_size_signature(instance):
    return tuple(sorted(len(piece) for piece in instance["pieces"]))


def _piece_shape_signature(instance):
    rel_pieces = [tuple(sorted(piece)) for piece in instance["pieces"]]
    return tuple(sorted(rel_pieces))


def build_diversity_report(instances, grid_size, requested_instances, seed, variant):
    key_counts = Counter()
    shape_counts = Counter()
    source_counts = Counter()
    source_unique_counts = Counter()
    piece_count_counts = Counter()
    size_signature_counts = Counter()
    piece_size_hist = Counter()

    for inst in instances:
        key = _canonical_instance_key(grid_size, inst["pieces"])
        shape_sig = _piece_shape_signature(inst)
        source = inst.get("instance_source", "unknown")

        key_counts[key] += 1
        shape_counts[shape_sig] += 1
        source_counts[source] += 1
        piece_count_counts[len(inst["pieces"])] += 1
        size_signature_counts[_piece_size_signature(inst)] += 1

        for piece in inst["pieces"]:
            piece_size_hist[len(piece)] += 1

    for inst in instances:
        key = _canonical_instance_key(grid_size, inst["pieces"])
        if key_counts[key] == 1:
            source_unique_counts[inst.get("instance_source", "unknown")] += 1

    duplicate_instances = sum(count - 1 for count in key_counts.values() if count > 1)
    unique_instances = len(key_counts)

    top_duplicate_groups = []
    for key, count in key_counts.most_common(10):
        if count <= 1:
            continue
        sample = next(inst for inst in instances if _canonical_instance_key(grid_size, inst["pieces"]) == key)
        top_duplicate_groups.append({
            "count": count,
            "instance_source": sample.get("instance_source", "unknown"),
            "piece_count": len(sample["pieces"]),
            "piece_sizes": sorted(len(piece) for piece in sample["pieces"]),
        })

    top_size_signatures = [
        {
            "count": count,
            "piece_count": len(sig),
            "piece_sizes": list(sig),
        }
        for sig, count in size_signature_counts.most_common(10)
    ]

    report = {
        "grid_size": grid_size,
        "requested_instances": requested_instances,
        "returned_instances": len(instances),
        "seed": seed,
        "constructive_variant": variant,
        "summary": {
            "unique_instances": unique_instances,
            "duplicate_instances": duplicate_instances,
            "unique_rate": unique_instances / max(1, len(instances)),
            "duplicate_rate": duplicate_instances / max(1, len(instances)),
            "unique_shape_signatures": len(shape_counts),
        },
        "source_breakdown": dict(sorted(source_counts.items())),
        "source_unique_breakdown": dict(sorted(source_unique_counts.items())),
        "piece_count_distribution": dict(sorted(piece_count_counts.items())),
        "piece_size_histogram": dict(sorted(piece_size_hist.items())),
        "top_size_signatures": top_size_signatures,
        "top_duplicate_groups": top_duplicate_groups,
        "notes": [
            "High duplicate_rate means the current constructive family is too repetitive for serious training.",
            "Partition-style constructive families can still be useful for deterministic test-case generation even when training diversity is weak.",
        ],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Audit constructive generator diversity.")
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--num-instances", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--constructive-variant", type=str, default="mixed")
    parser.add_argument("--allow-duplicate-fallback", action="store_true")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    instances = generate_constructive_puzzle_instances(
        num_instances=args.num_instances,
        grid_size=args.grid_size,
        seed=args.seed,
        large_suite_type=args.constructive_variant,
        verbose=False,
        allow_duplicate_fallback=args.allow_duplicate_fallback,
    )
    report = build_diversity_report(
        instances=instances,
        grid_size=args.grid_size,
        requested_instances=args.num_instances,
        seed=args.seed,
        variant=args.constructive_variant,
    )

    print("=" * 80)
    print("CONSTRUCTIVE DIVERSITY AUDIT")
    print("=" * 80)
    print(
        f"grid={args.grid_size} requested={args.num_instances} returned={report['returned_instances']} "
        f"variant={args.constructive_variant} dup_fallback={args.allow_duplicate_fallback}"
    )
    print(
        f"unique={report['summary']['unique_instances']} "
        f"duplicate_instances={report['summary']['duplicate_instances']} "
        f"unique_rate={report['summary']['unique_rate']:.3f} "
        f"duplicate_rate={report['summary']['duplicate_rate']:.3f}"
    )
    print(f"source_breakdown={report['source_breakdown']}")
    print(f"piece_count_distribution={report['piece_count_distribution']}")

    if report["top_duplicate_groups"]:
        print("top_duplicate_groups:")
        for row in report["top_duplicate_groups"][:5]:
            print(
                f"  count={row['count']} source={row['instance_source']} "
                f"piece_count={row['piece_count']} piece_sizes={row['piece_sizes']}"
            )
    else:
        print("top_duplicate_groups: none")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()
