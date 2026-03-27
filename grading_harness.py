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
from hybrid_solver import hybrid_solve
from phase2.nn_solver import solve_with_nn


Coord = Tuple[int, int, int]
Piece = List[Coord]
Solution = Dict[int, Iterable[Coord]]


@dataclass
class Case:
    case_id: str
    grid_size: int
    pieces: List[Piece]
    expected_solvable: bool


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
    model_name: str,
    beam_width: int,
    timeout_nn: float,
    timeout_dlx: float,
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
    if mode == "dlx":
        return dlx_solve(pieces, grid_size=grid_size, find_all=False)
    if mode == "nn":
        return solve_with_nn(
            pieces,
            grid_size=grid_size,
            model_name=model_name,
            beam_width=beam_width,
            timeout=timeout_nn,
            device="cpu",
        )
    raise ValueError("mode must be one of: hybrid, dlx, nn")


def run_proxy_grading(
    mode: str = "hybrid",
    model_name: str = "soma_3x3x3_quick",
    beam_width: int = 32,
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
