"""Run the 150-case A+ fixture against the current solver to establish a baseline.

Usage:
    python3 run_fixture_baseline.py --mode size_gated --per-case-timeout 20
    python3 run_fixture_baseline.py --subsample 3    # 3 per stratum smoke test

Output:
    fixtures/baseline_<mode>.json -- per-case results + aggregate score
    Console: human-readable report
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from fixture_a_plus import (
    DEFAULT_FIXTURE_PATH,
    load_fixture,
    print_fixture_report,
    run_fixture_grading,
)


def subsample_cases(cases, k_per_stratum, seed=561):
    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    for c in cases:
        by_stratum[c.stratum].append(c)
    out = []
    for stratum, rows in sorted(by_stratum.items()):
        rng.shuffle(rows)
        out.extend(rows[:k_per_stratum])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(DEFAULT_FIXTURE_PATH))
    ap.add_argument("--mode", default="size_gated",
                    choices=["hybrid", "size_gated", "dlx", "nn"])
    ap.add_argument("--model", default="soma_3x3x3_quick",
                    help="NN model name (used by hybrid/nn/size_gated inner solve)")
    ap.add_argument("--beam-width", type=int, default=32)
    ap.add_argument("--timeout-nn", type=float, default=8.0)
    ap.add_argument("--timeout-dlx", type=float, default=20.0)
    ap.add_argument("--per-case-timeout", type=float, default=30.0,
                    help="Hard wall-clock cap per case; min(timeout_nn, per-case) is used.")
    ap.add_argument("--subsample", type=int, default=0,
                    help="If >0, randomly sample this many cases per stratum.")
    ap.add_argument("--output", default="",
                    help="Output JSON path. Default: fixtures/baseline_<mode>.json")
    # Size-gated tuning flags
    ap.add_argument("--exact-only-max-grid", type=int, default=None,
                    help="Pure-DLX tier upper bound (default 4).")
    ap.add_argument("--exact-first-max-grid", type=int, default=None,
                    help="DLX-first tier upper bound (default 6). Bump to 9 to "
                         "route 7^3/9^3 through DLX.")
    ap.add_argument("--exact-first-timeout", type=float, default=None,
                    help="Per-case DLX budget in DLX-first tier (default 30s).")
    ap.add_argument("--large-allow-dlx", action="store_true",
                    help="Allow DLX fallback at the large-tier (>exact_first_max_grid) "
                         "if planners fail.")
    args = ap.parse_args()

    cases = load_fixture(Path(args.fixture))
    print(f"Loaded {len(cases)} cases from {args.fixture}")

    if args.subsample > 0:
        cases = subsample_cases(cases, args.subsample)
        print(f"Subsampled to {len(cases)} cases ({args.subsample}/stratum)")

    print(f"\nRunning solver mode={args.mode!r}, model={args.model!r}, "
          f"beam={args.beam_width}, per-case timeout={args.per_case_timeout}s")
    print(f"Estimated upper bound: {len(cases) * args.per_case_timeout / 60:.1f} min\n")

    extra = {}
    if args.exact_only_max_grid is not None:
        extra["exact_only_max_grid"] = args.exact_only_max_grid
    if args.exact_first_max_grid is not None:
        extra["exact_first_max_grid"] = args.exact_first_max_grid
    if args.exact_first_timeout is not None:
        extra["exact_first_timeout"] = args.exact_first_timeout
    if args.large_allow_dlx:
        extra["large_allow_dlx"] = True
    if extra:
        print(f"Solver overrides: {extra}\n")

    t0 = time.time()
    report = run_fixture_grading(
        cases,
        mode=args.mode,
        model_name=args.model,
        beam_width=args.beam_width,
        timeout_nn=args.timeout_nn,
        timeout_dlx=args.timeout_dlx,
        timeout_per_case=args.per_case_timeout,
        verbose=True,
        extra_solver_kwargs=extra,
    )
    wall = time.time() - t0
    print(f"\nTotal wall time: {wall:.1f}s ({wall/60:.2f} min)")

    print_fixture_report(report)

    out_path = Path(args.output) if args.output else (
        Path(args.fixture).parent / f"baseline_{args.mode}.json"
    )
    payload = {
        "config": {
            "mode": args.mode,
            "model_name": args.model,
            "beam_width": args.beam_width,
            "timeout_nn": args.timeout_nn,
            "timeout_dlx": args.timeout_dlx,
            "per_case_timeout": args.per_case_timeout,
            "subsample_per_stratum": args.subsample or None,
            "n_cases_run": len(cases),
            "wall_time_sec": wall,
            "extra_solver_kwargs": extra,
        },
        "score": report["score"],
        "results": report["results"],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
