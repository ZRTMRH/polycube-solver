"""
Proxy grading harness for Professor Cubazoid's 3D Tetris project.

What this does:
1) Builds a deterministic 20-case suite (mix of 2x2x2 and 3x3x3 cases)
2) Uses DLX as oracle to label each case as solvable/unsolvable
3) Runs a chosen solver and scores correctness
4) Reports whether the "15/20" A+ bar is met on this proxy suite
"""

from __future__ import annotations

import random
import time
import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from phase1.solver import solve as dlx_solve
from phase1.test_cases import (
    MONOMINO,
    DOMINO,
    L_TRICUBE,
    SOMA_PIECES,
    verify_solution,
)
from phase2.data_generator import enumerate_polycubes
from hybrid_solver import hybrid_solve, solve_size_gated
from phase2.nn_solver import solve_with_nn
from robust_generator import build_robust_constructive_case as _robust_generate


Coord = Tuple[int, int, int]
Piece = List[Coord]
Solution = Dict[int, Iterable[Coord]]


@dataclass
class Case:
    case_id: str
    grid_size: int
    pieces: List[Piece]
    expected_solvable: bool
    # Optional metadata used by the A+-target fixture. Defaults keep
    # backward compatibility with code that constructs Case positionally.
    stratum: str = ""
    generator: str = ""


@dataclass
class TierConfig:
    grid_size: int
    n_cases: int
    model_name: Optional[str] = None
    beam_width: Optional[int] = None
    timeout_nn: float = 30.0
    timeout_dlx: float = 90.0
    seed: int = 0
    exact_only_max_grid: int = 4
    exact_first_max_grid: int = 6
    exact_first_timeout: float = 30.0
    large_allow_dlx: bool = False
    allow_preplaced_fastpath: bool = False
    large_suite_type: str = "mixed"
    block_planner_enabled: bool = True
    block_timeout_dlx: float = 8.0
    block_timeout_nn: float = 0.0
    block_planner_trials: int = 3
    block_retries_per_block: int = 3


def _canonical_piece_key(piece: Piece) -> Tuple[Coord, ...]:
    return tuple(sorted(piece))


def _canonical_case_key(grid_size: int, pieces: List[Piece]) -> Tuple:
    parts = sorted(_canonical_piece_key(p) for p in pieces)
    return (grid_size, tuple(parts))


def _sample_piece_sizes(rng: random.Random, total: int = 27) -> Optional[List[int]]:
    """Sample piece sizes in {3,4,5} that sum to total."""
    sizes: List[int] = []
    remain = total
    for _ in range(25):
        choices = [s for s in (3, 4, 5) if remain - s >= 0]
        if not choices:
            return None
        s = rng.choice(choices)
        sizes.append(s)
        remain -= s
        if remain == 0:
            return sizes
    return None


def build_constructive_case(grid_size: int, seed: int) -> List[Piece]:
    """Construct one guaranteed-solvable case via unbiased greedy decomposition."""
    result = _robust_generate(grid_size, seed)
    if result is not None:
        return result
    raise RuntimeError(
        f"Failed to construct case for grid_size={grid_size} after retries."
    )


def build_constructive_suite(grid_size: int, n_cases: int, seed: int) -> List[Case]:
    """Build guaranteed-solvable benchmark cases using greedy decomposition."""
    return _build_robust_suite(grid_size, n_cases, seed)


def _build_robust_suite(grid_size: int, n_cases: int, seed: int) -> List[Case]:
    """Build n_cases using the unbiased greedy random generator."""
    from phase1.polycube import normalize

    rng = random.Random(seed)
    suite: List[Case] = []
    seen = set()

    attempts = 0
    max_attempts = max(200, n_cases * 50)
    while len(suite) < n_cases and attempts < max_attempts:
        attempts += 1
        local_seed = rng.randint(0, 10**9)
        pieces = _robust_generate(grid_size, local_seed)
        if pieces is None:
            continue
        # Normalize to origin (solver expects relative pieces)
        pieces = [
            list(normalize(frozenset(tuple(c) for c in p)))
            for p in pieces
        ]
        key = _canonical_case_key(grid_size, pieces)
        if key in seen:
            continue
        seen.add(key)
        suite.append(
            Case(
                case_id=f"greedy_{grid_size}_{len(suite):02d}",
                grid_size=grid_size,
                pieces=pieces,
                expected_solvable=True,
            )
        )

    if len(suite) < n_cases:
        raise RuntimeError(
            f"Could only build {len(suite)}/{n_cases} greedy cases "
            f"for grid {grid_size}."
        )
    return suite


def build_scale_suite(
    grid_size: int,
    n_cases: int,
    seed: int,
    dlx_timeout: float = 20.0,
    large_suite_type: str = "mixed",
) -> Tuple[List[Case], str]:
    """Build benchmark suite using unbiased greedy decomposition at all sizes."""
    suite = _build_robust_suite(grid_size, n_cases, seed)
    return suite, "greedy_random"


def _oracle_is_solvable(pieces: List[Piece], grid_size: int) -> bool:
    buf = io.StringIO()
    with redirect_stdout(buf):
        sols = dlx_solve(pieces, grid_size=grid_size, find_all=False)
    return len(sols) > 0


def build_proxy_suite(
    seed: int = 561,
    n_cases: int = 20,
    include_fixed_cases: bool = True,
) -> List[Case]:
    """Build deterministic proxy test cases to estimate grading performance."""
    if n_cases < 8 and include_fixed_cases:
        raise ValueError("n_cases must be >= 8 when include_fixed_cases=True")
    if n_cases < 1:
        raise ValueError("n_cases must be >= 1")

    rng = random.Random(seed)
    suite: List[Case] = []
    seen = set()

    # Guaranteed small baseline cases
    if include_fixed_cases:
        fixed = [
            ("fixed_2x2_monomino", 2, [MONOMINO] * 8),
            ("fixed_2x2_domino", 2, [DOMINO] * 4),
            ("fixed_2x2_impossible", 2, [DOMINO] * 3),
            ("fixed_2x2_ltri_plus_mono", 2, [L_TRICUBE] + [MONOMINO] * 5),
        ]
        for cid, g, pieces in fixed:
            key = _canonical_case_key(g, pieces)
            if key in seen:
                continue
            seen.add(key)
            suite.append(
                Case(
                    case_id=cid,
                    grid_size=g,
                    pieces=[list(p) for p in pieces],
                    expected_solvable=_oracle_is_solvable([list(p) for p in pieces], g),
                )
            )

    # Add several Soma permutations (known 3x3 solvable style)
    soma_target = 4 if include_fixed_cases else min(4, n_cases)
    for i in range(soma_target):
        perm = list(range(len(SOMA_PIECES)))
        rng.shuffle(perm)
        pieces = [SOMA_PIECES[j] for j in perm]
        key = _canonical_case_key(3, pieces)
        if key in seen:
            continue
        seen.add(key)
        suite.append(
            Case(
                case_id=f"soma_perm_{i}",
                grid_size=3,
                pieces=[list(p) for p in pieces],
                expected_solvable=True,
            )
        )

    # Random 3x3 cases using free polycubes of size 3..5
    catalog = enumerate_polycubes(max_size=5)
    by_size = {s: [list(poly) for poly in polys] for s, polys in catalog.items()}
    attempts = 0
    max_attempts = 5000

    while len(suite) < n_cases and attempts < max_attempts:
        attempts += 1
        sizes = _sample_piece_sizes(rng, total=27)
        if not sizes:
            continue

        pieces: List[Piece] = []
        ok = True
        for s in sizes:
            pool = by_size.get(s, [])
            if not pool:
                ok = False
                break
            pieces.append(rng.choice(pool))
        if not ok:
            continue

        key = _canonical_case_key(3, pieces)
        if key in seen:
            continue
        seen.add(key)

        expected = _oracle_is_solvable(pieces, 3)
        suite.append(
            Case(
                case_id=f"rand3_{len(suite):02d}",
                grid_size=3,
                pieces=[list(p) for p in pieces],
                expected_solvable=expected,
            )
        )

    if len(suite) < n_cases:
        raise RuntimeError(
            f"Could only build {len(suite)}/{n_cases} cases after {attempts} attempts."
        )

    return suite[:n_cases]


def _extract_solution_obj(raw_output):
    """Normalize solver output into a solution dict or None."""
    if raw_output is None:
        return None
    if isinstance(raw_output, dict) and "solution" in raw_output:
        return raw_output["solution"]
    if isinstance(raw_output, list):
        return raw_output[0] if raw_output else None
    if isinstance(raw_output, dict):
        return raw_output
    return None


def _default_solver(
    pieces: List[Piece],
    grid_size: int,
    mode: str,
    model_name: Optional[str],
    beam_width: int,
    timeout_nn: float,
    timeout_dlx: float,
    exact_only_max_grid: int = 4,
    exact_first_max_grid: int = 6,
    exact_first_timeout: float = 30.0,
    large_allow_dlx: bool = False,
    allow_preplaced_fastpath: bool = False,
    block_planner_enabled: bool = True,
    block_timeout_dlx: float = 8.0,
    block_timeout_nn: float = 0.0,
    block_planner_trials: int = 3,
    block_retries_per_block: int = 3,
):
    if mode == "hybrid":
        return hybrid_solve(
            pieces,
            grid_size=grid_size,
            model_name=model_name,
            beam_width=beam_width,
            timeout_nn=timeout_nn,
            timeout_dlx=timeout_dlx,
            device="cpu",
            verbose=False,
        )
    if mode == "size_gated":
        return solve_size_gated(
            pieces,
            grid_size=grid_size,
            model_name=model_name,
            beam_width=beam_width,
            timeout_nn=timeout_nn,
            timeout_dlx=timeout_dlx,
            device="cpu",
            verbose=False,
            exact_only_max_grid=exact_only_max_grid,
            exact_first_max_grid=exact_first_max_grid,
            exact_first_timeout=exact_first_timeout,
            large_allow_dlx=large_allow_dlx,
            allow_preplaced_fastpath=allow_preplaced_fastpath,
            block_planner_enabled=block_planner_enabled,
            block_timeout_dlx=block_timeout_dlx,
            block_timeout_nn=block_timeout_nn,
            block_planner_trials=block_planner_trials,
            block_retries_per_block=block_retries_per_block,
        )
    if mode == "dlx":
        return dlx_solve(pieces, grid_size=grid_size, find_all=False)
    if mode == "nn":
        if not model_name:
            raise ValueError("mode='nn' requires a model_name")
        return solve_with_nn(
            pieces,
            grid_size=grid_size,
            model_name=model_name,
            beam_width=beam_width,
            timeout=timeout_nn,
            device="cpu",
        )
    raise ValueError("mode must be one of: hybrid, size_gated, dlx, nn")


def run_proxy_grading(
    mode: str = "hybrid",
    model_name: str = "soma_3x3x3_quick",
    beam_width: Optional[int] = None,
    timeout_nn: float = 8.0,
    timeout_dlx: float = 20.0,
    seed: int = 561,
    n_cases: int = 20,
    solver_fn: Optional[Callable[[List[Piece], int], object]] = None,
    include_fixed_cases: bool = True,
):
    """Run the 20-case proxy grading evaluation.

    Returns:
        Dict with summary and per-case results.
    """
    suite = build_proxy_suite(
        seed=seed,
        n_cases=n_cases,
        include_fixed_cases=include_fixed_cases,
    )
    results = []
    correct = 0

    for case in suite:
        t0 = time.time()
        if solver_fn is None:
            raw = _default_solver(
                case.pieces,
                case.grid_size,
                mode=mode,
                model_name=model_name,
                beam_width=beam_width,
                timeout_nn=timeout_nn,
                timeout_dlx=timeout_dlx,
            )
        else:
            raw = solver_fn(case.pieces, case.grid_size)
        elapsed = time.time() - t0

        predicted = _extract_solution_obj(raw)
        predicted_solvable = predicted is not None

        is_correct = False
        reason = ""
        if case.expected_solvable:
            if predicted is None:
                reason = "missed solvable case (returned None)"
            else:
                is_valid = verify_solution(predicted, case.grid_size)
                if is_valid:
                    is_correct = True
                else:
                    reason = "returned invalid solution"
        else:
            if predicted is None:
                is_correct = True
            else:
                reason = "returned solution for unsolvable case"

        if is_correct:
            correct += 1

        results.append(
            {
                "case_id": case.case_id,
                "grid_size": case.grid_size,
                "num_pieces": len(case.pieces),
                "expected_solvable": case.expected_solvable,
                "predicted_solvable": predicted_solvable,
                "correct": is_correct,
                "time_sec": elapsed,
                "reason": reason,
            }
        )

    score = {
        "correct": correct,
        "total": len(results),
        "passes_a_plus_bar": correct >= 15,
        "accuracy": correct / max(1, len(results)),
        "avg_time_sec": sum(r["time_sec"] for r in results) / max(1, len(results)),
        "p95_time_sec": sorted(r["time_sec"] for r in results)[int(0.95 * (len(results) - 1))],
    }
    solvable_rows = [r for r in results if r["expected_solvable"]]
    unsolvable_rows = [r for r in results if not r["expected_solvable"]]
    if solvable_rows:
        score["solvable_correct"] = sum(1 for r in solvable_rows if r["correct"])
        score["solvable_total"] = len(solvable_rows)
        score["solvable_accuracy"] = score["solvable_correct"] / len(solvable_rows)
    else:
        score["solvable_correct"] = 0
        score["solvable_total"] = 0
        score["solvable_accuracy"] = 0.0
    if unsolvable_rows:
        score["unsolvable_correct"] = sum(1 for r in unsolvable_rows if r["correct"])
        score["unsolvable_total"] = len(unsolvable_rows)
        score["unsolvable_accuracy"] = score["unsolvable_correct"] / len(unsolvable_rows)
    else:
        score["unsolvable_correct"] = 0
        score["unsolvable_total"] = 0
        score["unsolvable_accuracy"] = 0.0

    return {"score": score, "results": results}


def run_multi_seed_proxy_grading(
    seeds: List[int],
    mode: str = "hybrid",
    model_name: str = "soma_3x3x3_quick",
    beam_width: int = 32,
    timeout_nn: float = 8.0,
    timeout_dlx: float = 20.0,
    n_cases: int = 20,
    include_fixed_cases: bool = True,
):
    """Run proxy grading for multiple seeds and aggregate summary stats."""
    reports = []
    for seed in seeds:
        rep = run_proxy_grading(
            mode=mode,
            model_name=model_name,
            beam_width=beam_width,
            timeout_nn=timeout_nn,
            timeout_dlx=timeout_dlx,
            seed=seed,
            n_cases=n_cases,
            include_fixed_cases=include_fixed_cases,
        )
        reports.append({"seed": seed, "report": rep})

    scores = [x["report"]["score"] for x in reports]
    total_correct = sum(s["correct"] for s in scores)
    total_cases = sum(s["total"] for s in scores)
    aggregate = {
        "num_seeds": len(seeds),
        "mean_accuracy": sum(s["accuracy"] for s in scores) / max(1, len(scores)),
        "min_accuracy": min(s["accuracy"] for s in scores),
        "max_accuracy": max(s["accuracy"] for s in scores),
        "mean_correct": sum(s["correct"] for s in scores) / max(1, len(scores)),
        "aggregate_correct": total_correct,
        "aggregate_total": total_cases,
        "aggregate_accuracy": total_correct / max(1, total_cases),
        "mean_solvable_accuracy": sum(s["solvable_accuracy"] for s in scores) / max(1, len(scores)),
        "mean_unsolvable_accuracy": sum(s["unsolvable_accuracy"] for s in scores) / max(1, len(scores)),
    }
    return {"aggregate": aggregate, "runs": reports}


def _default_model_for_grid(grid_size: int) -> Optional[str]:
    if grid_size == 3:
        return "soma_3x3x3_quick"
    if grid_size == 4:
        return "4x4x4_adi2"
    return None


def run_scale_tier_benchmark(
    tiers: Optional[List[TierConfig]] = None,
    mode: str = "hybrid",
):
    """Run benchmark tiers for progressively larger cubes.

    Each tier uses constructive guaranteed-solvable cases and reports:
    - valid solve rate
    - average/p95 runtime
    - method usage breakdown (nn/dlx/none)
    - submethod usage breakdown (beam/retry/complete_fallback/dlx_*)
    """
    if tiers is None:
        tiers = [
            TierConfig(grid_size=4, n_cases=2, seed=404, timeout_nn=30.0, timeout_dlx=90.0),
            TierConfig(grid_size=5, n_cases=1, seed=505, timeout_nn=45.0, timeout_dlx=120.0),
            TierConfig(grid_size=6, n_cases=1, seed=606, timeout_nn=60.0, timeout_dlx=180.0),
        ]

    results = []
    for tier in tiers:
        model_name = tier.model_name if tier.model_name is not None else _default_model_for_grid(tier.grid_size)
        suite, suite_source = build_scale_suite(
            grid_size=tier.grid_size,
            n_cases=tier.n_cases,
            seed=tier.seed,
            dlx_timeout=tier.timeout_dlx,
            large_suite_type=tier.large_suite_type,
        )

        row_results = []
        valid_solved = 0
        method_counts = {"nn": 0, "dlx": 0, "planner": 0, "none": 0, "other": 0}
        submethod_counts = {}

        for case in suite:
            t0 = time.time()
            raw = _default_solver(
                case.pieces,
                case.grid_size,
                mode=mode,
                model_name=model_name,
                beam_width=tier.beam_width,
                timeout_nn=tier.timeout_nn,
                timeout_dlx=tier.timeout_dlx,
                exact_only_max_grid=tier.exact_only_max_grid,
                exact_first_max_grid=tier.exact_first_max_grid,
                exact_first_timeout=tier.exact_first_timeout,
                large_allow_dlx=tier.large_allow_dlx,
                allow_preplaced_fastpath=tier.allow_preplaced_fastpath,
                block_planner_enabled=tier.block_planner_enabled,
                block_timeout_dlx=tier.block_timeout_dlx,
                block_timeout_nn=tier.block_timeout_nn,
                block_planner_trials=tier.block_planner_trials,
                block_retries_per_block=tier.block_retries_per_block,
            )
            elapsed = time.time() - t0

            predicted = _extract_solution_obj(raw)
            solved = predicted is not None
            valid = verify_solution(predicted, tier.grid_size) if solved else False
            if valid:
                valid_solved += 1

            method = "none"
            submethod = "none"
            controller = ""
            route_tier = ""
            planner_diag = None
            if isinstance(raw, dict):
                raw_method = raw.get("method")
                raw_submethod = raw.get("submethod")
                if raw_method is not None:
                    method = raw_method
                if raw_submethod:
                    submethod = raw_submethod
                controller = raw.get("controller") or ""
                route_tier = raw.get("tier") or ""
                planner_diag = raw.get("planner_diag")
            elif mode == "nn":
                method = "nn"
                submethod = "beam"
            elif mode == "dlx":
                method = "dlx"
                submethod = "dlx_exact"

            if method in method_counts:
                method_counts[method] += 1
            else:
                method_counts["other"] += 1
            submethod_counts[submethod] = submethod_counts.get(submethod, 0) + 1

            row_results.append(
                {
                    "case_id": case.case_id,
                    "num_pieces": len(case.pieces),
                    "solved": solved,
                    "valid": valid,
                    "method": method,
                    "submethod": submethod,
                    "controller": controller if controller else "none",
                    "route_tier": route_tier if route_tier else "none",
                    "planner_diag": planner_diag,
                    "time_sec": elapsed,
                }
            )

        times = sorted(r["time_sec"] for r in row_results)
        p95 = times[int(0.95 * (len(times) - 1))] if times else 0.0
        tier_summary = {
            "grid_size": tier.grid_size,
            "n_cases": tier.n_cases,
            "model_name": model_name,
            "mode": mode,
            "suite_source": suite_source,
            "valid_solved": valid_solved,
            "solve_rate": valid_solved / max(1, tier.n_cases),
            "avg_time_sec": sum(times) / max(1, len(times)),
            "p95_time_sec": p95,
            "method_counts": method_counts,
            "submethod_counts": submethod_counts,
            "cases": row_results,
        }
        results.append(tier_summary)

    return {"tiers": results}


def print_proxy_grading_report(report):
    score = report["score"]
    print("=" * 72)
    print("Proxy Grading Report")
    print("=" * 72)
    print(
        f"Correct: {score['correct']}/{score['total']}  "
        f"Accuracy: {score['accuracy']:.3f}  "
        f"A+ bar (>=15/20): {score['passes_a_plus_bar']}"
    )
    print(
        f"Solvable accuracy: {score['solvable_correct']}/{score['solvable_total']} "
        f"({score['solvable_accuracy']:.3f})  "
        f"Unsolvable accuracy: {score['unsolvable_correct']}/{score['unsolvable_total']} "
        f"({score['unsolvable_accuracy']:.3f})"
    )
    print(
        f"Avg time: {score['avg_time_sec']:.3f}s  "
        f"P95 time: {score['p95_time_sec']:.3f}s"
    )
    print("-" * 72)
    for r in report["results"]:
        status = "PASS" if r["correct"] else "FAIL"
        note = f" ({r['reason']})" if r["reason"] else ""
        print(
            f"{status:4} {r['case_id']:18} "
            f"grid={r['grid_size']} pieces={r['num_pieces']:2d} "
            f"exp={str(r['expected_solvable']):5} pred={str(r['predicted_solvable']):5} "
            f"time={r['time_sec']:.3f}s{note}"
        )
    print("=" * 72)


def print_multi_seed_report(multi_report):
    agg = multi_report["aggregate"]
    print("=" * 72)
    print("Multi-Seed Proxy Grading Summary")
    print("=" * 72)
    print(
        f"Seeds: {agg['num_seeds']}  "
        f"Mean acc: {agg['mean_accuracy']:.3f}  "
        f"Min/Max acc: {agg['min_accuracy']:.3f}/{agg['max_accuracy']:.3f}"
    )
    print(
        f"Aggregate correct: {agg['aggregate_correct']}/{agg['aggregate_total']} "
        f"({agg['aggregate_accuracy']:.3f})"
    )
    print(
        f"Mean solvable acc: {agg['mean_solvable_accuracy']:.3f}  "
        f"Mean unsolvable acc: {agg['mean_unsolvable_accuracy']:.3f}"
    )
    print("-" * 72)
    for row in multi_report["runs"]:
        s = row["report"]["score"]
        print(
            f"seed={row['seed']}  "
            f"correct={s['correct']}/{s['total']}  "
            f"acc={s['accuracy']:.3f}  "
            f"solv={s['solvable_accuracy']:.3f}  "
            f"unsolv={s['unsolvable_accuracy']:.3f}"
        )
    print("=" * 72)


def print_scale_tier_report(scale_report):
    print("=" * 80)
    print("Scale Tier Benchmark")
    print("=" * 80)
    for tier in scale_report["tiers"]:
        print(
            f"Grid {tier['grid_size']}^3 | cases={tier['n_cases']} | "
            f"valid_solved={tier['valid_solved']} | solve_rate={tier['solve_rate']:.3f}"
        )
        print(
            f"  avg={tier['avg_time_sec']:.3f}s p95={tier['p95_time_sec']:.3f}s "
            f"model={tier['model_name']} mode={tier['mode']} suite={tier['suite_source']}"
        )
        mc = tier["method_counts"]
        print(
            "  method_counts: "
            f"nn={mc['nn']} dlx={mc['dlx']} planner={mc['planner']} "
            f"none={mc['none']} other={mc['other']}"
        )
        smc = tier.get("submethod_counts", {})
        if smc:
            smc_items = " ".join(f"{k}={v}" for k, v in sorted(smc.items()))
            print(f"  submethod_counts: {smc_items}")
        for row in tier["cases"]:
            diag_txt = ""
            if isinstance(row.get("planner_diag"), dict):
                btot = row["planner_diag"].get("blocks_total")
                bsol = row["planner_diag"].get("blocks_solved")
                rsn = row["planner_diag"].get("reason")
                diag_txt = f" blocks={bsol}/{btot} reason={rsn}"
            print(
                f"    case={row['case_id']} pieces={row['num_pieces']:2d} "
                f"method={row['method']:4} submethod={row.get('submethod', 'none')} "
                f"controller={row.get('controller', 'none')} "
                f"tier={row.get('route_tier', 'none')} "
                f"solved={row['solved']} valid={row['valid']} "
                f"time={row['time_sec']:.3f}s{diag_txt}"
            )
        print("-" * 80)
    print("=" * 80)


if __name__ == "__main__":
    rep = run_proxy_grading(
        mode="hybrid",
        model_name="soma_3x3x3_quick",
        beam_width=32,
        timeout_nn=8.0,
        timeout_dlx=20.0,
        seed=561,
        n_cases=20,
    )
    print_proxy_grading_report(rep)
