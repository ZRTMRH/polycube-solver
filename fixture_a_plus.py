"""
A+-target fixture: 150 stratified cases for estimating solver performance
against the instructor's unseen 20-case grading suite.

Rationale
---------
The instructor runs 20 hidden cases and grants A+ for >=15 correct. With
only 20 cases and an 85% true solve rate, the A+ probability is ~0.93;
with 80% it drops to ~0.80. We therefore need to estimate our solver's
true rate with enough precision to distinguish "safely above 85%" from
"marginal." A single 20-case pass has SE ~8%; 150 cases gets SE ~3-4%.

Stratification
--------------
Grid-size distribution the grader uses is secret; the fixture spans
3^3..12^3. Unsolvable cases are known to be in the mix (confirmed with
the user), so the fixture is ~2:1 solvable:unsolvable.

For 12^3 (the target size) three constructive generators are used to
catch between-generator variance — a poor proxy for true grader
uncertainty but the best we can do without seeing the grader's spec.

Unsolvable generation uses a single mechanism (volume_mismatch): mutate
a solvable constructive case by dropping or duplicating a piece so the
total cell count differs from N^3. This is trivially unsolvable by
cardinality and detectable by any sane solver. We knowingly miss
grader cases that are unsolvable for subtler reasons (tight
adversarial sets, parity violations) -- the fixture therefore likely
overstates our unsolvable-detection rate.

Output
------
fixtures/a_plus_150.json -- deterministic, keyed from seed=561.
Reload with load_fixture().
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import List

from grading_harness import (
    Case,
    build_constructive_case,
    build_constructive_suite,
    build_mixed_constructive_suite,
    build_striped_constructive_suite,
    _canonical_case_key,
)
from phase2.data_generator import enumerate_polycubes, generate_puzzle_instances


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
DEFAULT_FIXTURE_PATH = FIXTURE_DIR / "a_plus_150.json"
DEFAULT_SEED = 561


# ── Unsolvable generators ───────────────────────────────────────────────────

def _mutate_volume_drop(pieces, rng):
    """Drop one random piece. Volume < N^3 --> trivially unsolvable."""
    idx = rng.randrange(len(pieces))
    return [p for i, p in enumerate(pieces) if i != idx]


def _mutate_volume_excess(pieces, rng):
    """Duplicate one piece. Volume > N^3 --> trivially unsolvable."""
    idx = rng.randrange(len(pieces))
    return pieces + [list(pieces[idx])]


def build_unsolvable_from_bases(
    grid_size: int, n_cases: int, base_cases: List[Case], seed: int
) -> List[Case]:
    """Mutate given solvable base cases into volume-mismatch unsolvables.

    Decoupling from base-case generation lets us reuse whatever generator
    works at this grid size (constructive at 3^3, DLX-oracled at 4^3/5^3,
    mixed/striped at 7^3+).
    """
    if not base_cases:
        raise ValueError("Need at least one base case to mutate.")
    rng = random.Random(seed)
    cases: List[Case] = []
    seen = set()
    attempts = 0
    max_attempts = max(100, n_cases * 30)

    while len(cases) < n_cases and attempts < max_attempts:
        attempts += 1
        base = rng.choice(base_cases).pieces
        mode = rng.choice(("drop", "excess"))
        mutated = (
            _mutate_volume_drop(base, rng)
            if mode == "drop"
            else _mutate_volume_excess(base, rng)
        )

        if sum(len(p) for p in mutated) == grid_size ** 3:
            continue

        key = _canonical_case_key(grid_size, mutated)
        if key in seen:
            continue
        seen.add(key)

        cases.append(
            Case(
                case_id=f"unsolv_{mode}_{grid_size}_{len(cases):02d}",
                grid_size=grid_size,
                pieces=mutated,
                expected_solvable=False,
                stratum=f"{grid_size}^3_unsolvable",
                generator=f"volume_{mode}",
            )
        )

    if len(cases) < n_cases:
        raise RuntimeError(
            f"Only built {len(cases)}/{n_cases} volume-mismatch cases at grid {grid_size}"
        )
    return cases


# ── Solvable sub-suite builders (thin adapters that tag stratum/generator) ──

def _tag(cases: List[Case], stratum: str, generator: str) -> List[Case]:
    """Mutate-in-place tag cases with stratum/generator (safe since they're freshly built)."""
    for c in cases:
        c.stratum = stratum
        c.generator = generator
    return cases


def _solvable_constructive(grid_size: int, n: int, seed: int) -> List[Case]:
    cases = build_constructive_suite(grid_size, n, seed)
    return _tag(cases, f"{grid_size}^3_solvable", "constructive")


def _solvable_mixed(grid_size: int, n: int, seed: int) -> List[Case]:
    cases = build_mixed_constructive_suite(grid_size, n, seed)
    return _tag(cases, f"{grid_size}^3_solvable", "mixed_constructive")


def _solvable_striped(grid_size: int, n: int, seed: int) -> List[Case]:
    cases = build_striped_constructive_suite(grid_size, n, seed)
    return _tag(cases, f"{grid_size}^3_solvable", "striped_constructive")


_POLYCUBE_CATALOG_CACHE = None


def _solvable_dlx(grid_size: int, n: int, seed: int) -> List[Case]:
    """DLX-oracled random solvable cases. Works at small grids (<=5) where
    build_constructive_case hits topological dead-ends."""
    global _POLYCUBE_CATALOG_CACHE
    if _POLYCUBE_CATALOG_CACHE is None:
        _POLYCUBE_CATALOG_CACHE = enumerate_polycubes(max_size=5)

    from contextlib import redirect_stdout
    import io as _io
    buf = _io.StringIO()
    with redirect_stdout(buf):  # suppress verbose DLX chatter
        instances = generate_puzzle_instances(
            num_instances=n,
            grid_size=grid_size,
            polycube_catalog=_POLYCUBE_CATALOG_CACHE,
            min_piece_size=3,
            max_piece_size=5,
            dlx_timeout=20.0,
            seed=seed,
            verbose=False,
        )

    cases = [
        Case(
            case_id=f"dlx_{grid_size}_{i:02d}",
            grid_size=grid_size,
            pieces=inst["pieces"],
            expected_solvable=True,
            stratum=f"{grid_size}^3_solvable",
            generator="dlx_random",
        )
        for i, inst in enumerate(instances)
    ]
    if len(cases) < n:
        raise RuntimeError(
            f"DLX generator only produced {len(cases)}/{n} solvable cases at {grid_size}^3"
        )
    return cases


# ── The 150-case fixture ────────────────────────────────────────────────────

# Per-grid generator choices are pragmatic:
# - build_constructive_case: reliable at 3^3, flaky at 4^3, broken at 5^3
#   (grow-connected-piece hits topological dead ends as N grows).
# - build_mixed_constructive_case & striped: designed for large grids,
#   low diversity at N<=5.
# - DLX-based generate_puzzle_instances: works at 3^3..5^3 (DLX runs in
#   reasonable time); too slow at 7^3+ to rely on.
# So: DLX for small grids, mixed/striped/constructive for large grids.
FIXTURE_SPEC = [
    (3,  [(10, _solvable_dlx)],                       5),
    (4,  [(15, _solvable_dlx)],                       5),
    (5,  [(15, _solvable_dlx)],                       5),
    (7,  [(15, _solvable_mixed)],                     5),
    (9,  [(15, _solvable_mixed)],                     5),
    (12, [(20, _solvable_mixed),
          (20, _solvable_striped)],                  15),
]


def build_a_plus_fixture(seed: int = DEFAULT_SEED) -> List[Case]:
    """Assemble the full 150-case stratified fixture."""
    rng = random.Random(seed)
    all_cases: List[Case] = []

    for grid_size, solvable_specs, n_unsolvable in FIXTURE_SPEC:
        grid_solvables: List[Case] = []
        for n, builder in solvable_specs:
            sub_seed = rng.randint(0, 10**9)
            grid_solvables.extend(builder(grid_size, n, sub_seed))
        all_cases.extend(grid_solvables)

        if n_unsolvable > 0:
            sub_seed = rng.randint(0, 10**9)
            # Mutate just-built solvable cases at this grid size — guarantees
            # we have valid base pieces regardless of which generator succeeds.
            all_cases.extend(
                build_unsolvable_from_bases(
                    grid_size, n_unsolvable, grid_solvables, sub_seed
                )
            )

    return all_cases


# ── Serialization ───────────────────────────────────────────────────────────

def _case_to_dict(case: Case) -> dict:
    # Pieces are lists of (x,y,z) tuples; json-serialize as lists-of-lists.
    return {
        "case_id": case.case_id,
        "grid_size": case.grid_size,
        "pieces": [[list(cell) for cell in piece] for piece in case.pieces],
        "expected_solvable": case.expected_solvable,
        "stratum": case.stratum,
        "generator": case.generator,
    }


def _case_from_dict(d: dict) -> Case:
    return Case(
        case_id=d["case_id"],
        grid_size=d["grid_size"],
        pieces=[[tuple(cell) for cell in piece] for piece in d["pieces"]],
        expected_solvable=d["expected_solvable"],
        stratum=d.get("stratum", ""),
        generator=d.get("generator", ""),
    )


def save_fixture(cases: List[Case], path: Path = DEFAULT_FIXTURE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "seed": DEFAULT_SEED,
        "n_cases": len(cases),
        "cases": [_case_to_dict(c) for c in cases],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_fixture(path: Path = DEFAULT_FIXTURE_PATH) -> List[Case]:
    path = Path(path)
    data = json.loads(path.read_text())
    return [_case_from_dict(d) for d in data["cases"]]


# ── Running solver against fixture + scoring ────────────────────────────────

def _p_aplus_given_phat(p_hat: float, n_grader: int = 20, threshold: int = 15) -> float:
    """P(Binomial(n_grader, p_hat) >= threshold). No scipy dependency."""
    from math import comb
    if p_hat <= 0:
        return 0.0
    if p_hat >= 1:
        return 1.0
    total = 0.0
    q = 1.0 - p_hat
    for k in range(threshold, n_grader + 1):
        total += comb(n_grader, k) * (p_hat ** k) * (q ** (n_grader - k))
    return total


def run_fixture_grading(
    cases: List[Case],
    mode: str = "hybrid",
    model_name: str = "soma_3x3x3_quick",
    beam_width: int = 32,
    timeout_nn: float = 8.0,
    timeout_dlx: float = 20.0,
    timeout_per_case: float = 60.0,
    verbose: bool = True,
    extra_solver_kwargs: dict | None = None,
):
    """Run the solver against every fixture case. Score correctness.

    Timeout policy (per agreed A+ strategy): solution=None is treated as
    a claim of 'unsolvable'. We log the submethod so we can see whether
    each 'unsolvable' claim was proved (DLX exhausted) or guessed (timed out).

    Returns a dict: {'results': [...], 'score': {...}}.
    """
    import time
    from grading_harness import _default_solver, _extract_solution_obj, verify_solution

    extra_solver_kwargs = extra_solver_kwargs or {}

    results = []
    for i, case in enumerate(cases):
        if verbose and i % 10 == 0:
            print(f"  [{i:3d}/{len(cases)}] {case.case_id} ({case.stratum})...")
        t0 = time.time()
        try:
            raw = _default_solver(
                case.pieces,
                case.grid_size,
                mode=mode,
                model_name=model_name,
                beam_width=beam_width,
                timeout_nn=min(timeout_nn, timeout_per_case),
                timeout_dlx=min(timeout_dlx, timeout_per_case),
                **extra_solver_kwargs,
            )
        except Exception as e:
            raw = {"solution": None, "method": None,
                   "submethod": f"solver_error:{type(e).__name__}"}
        elapsed = time.time() - t0

        predicted = _extract_solution_obj(raw)
        submethod = raw.get("submethod", "") if isinstance(raw, dict) else ""
        predicted_solvable = predicted is not None

        # Proof status for "unsolvable" claims:
        #   proved    -- DLX completed and returned None (no_solution)
        #   guessed   -- timed out or errored; we default to unsolvable
        if predicted is None:
            if "no_solution" in submethod:
                proof_status = "proved_unsolvable"
            elif "timeout" in submethod or "error" in submethod:
                proof_status = "guessed_unsolvable_on_timeout"
            else:
                proof_status = "claimed_unsolvable"
        else:
            proof_status = "solution_produced"

        # Score correctness
        is_correct = False
        reason = ""
        if case.expected_solvable:
            if predicted is None:
                reason = f"missed solvable ({proof_status})"
            elif verify_solution(predicted, case.grid_size):
                is_correct = True
            else:
                reason = "returned invalid solution"
        else:
            if predicted is None:
                is_correct = True
            else:
                reason = "returned solution for unsolvable case"

        results.append({
            "case_id": case.case_id,
            "grid_size": case.grid_size,
            "stratum": case.stratum,
            "generator": case.generator,
            "num_pieces": len(case.pieces),
            "expected_solvable": case.expected_solvable,
            "predicted_solvable": predicted_solvable,
            "correct": is_correct,
            "time_sec": elapsed,
            "submethod": submethod,
            "proof_status": proof_status,
            "reason": reason,
        })

    return {"results": results, "score": score_fixture(results)}


def score_fixture(results: List[dict]) -> dict:
    """Compute aggregate and per-stratum metrics."""
    from collections import Counter

    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    p_hat = correct / max(1, n)

    # Per-stratum
    strata = sorted({r["stratum"] for r in results})
    per_stratum = {}
    for s in strata:
        rows = [r for r in results if r["stratum"] == s]
        per_stratum[s] = {
            "n": len(rows),
            "correct": sum(1 for r in rows if r["correct"]),
            "accuracy": sum(1 for r in rows if r["correct"]) / len(rows),
            "avg_time": sum(r["time_sec"] for r in rows) / len(rows),
        }

    # Solvable / unsolvable breakdown
    solvable = [r for r in results if r["expected_solvable"]]
    unsolvable = [r for r in results if not r["expected_solvable"]]

    # Unsolvable detection breakdown: proved vs guessed
    proof_counts = Counter(r["proof_status"] for r in results if not r["predicted_solvable"])

    min_stratum_acc = min((s["accuracy"] for s in per_stratum.values()), default=0.0)
    worst_stratum = min(per_stratum.items(), key=lambda kv: kv[1]["accuracy"])[0] if per_stratum else ""

    return {
        "n": n,
        "correct": correct,
        "p_hat": p_hat,
        "p_a_plus": _p_aplus_given_phat(p_hat),
        "solvable_accuracy": sum(1 for r in solvable if r["correct"]) / max(1, len(solvable)),
        "solvable_n": len(solvable),
        "unsolvable_accuracy": sum(1 for r in unsolvable if r["correct"]) / max(1, len(unsolvable)),
        "unsolvable_n": len(unsolvable),
        "proof_breakdown": dict(proof_counts),
        "per_stratum": per_stratum,
        "min_stratum_accuracy": min_stratum_acc,
        "worst_stratum": worst_stratum,
        "avg_time_sec": sum(r["time_sec"] for r in results) / max(1, n),
        "p95_time_sec": (sorted(r["time_sec"] for r in results)[int(0.95 * (n - 1))]
                         if n > 0 else 0.0),
    }


def print_fixture_report(report: dict) -> None:
    """Human-readable summary of run_fixture_grading output."""
    s = report["score"]
    print()
    print("=" * 72)
    print("A+ Fixture Grading Report")
    print("=" * 72)
    print(f"Aggregate: {s['correct']}/{s['n']}   p_hat = {s['p_hat']:.3f}")
    print(f"P(A+) on the instructor's 20-case test: {s['p_a_plus']:.3f}")
    print(f"  (computed from p_hat; target >= 0.90 for comfortable A+)")
    print()
    print(f"Solvable:   {s['solvable_accuracy']:.3f} across {s['solvable_n']} cases")
    print(f"Unsolvable: {s['unsolvable_accuracy']:.3f} across {s['unsolvable_n']} cases")
    print()
    print("Proof status of 'unsolvable' claims:")
    for k, v in sorted(s["proof_breakdown"].items()):
        print(f"  {k:35s} {v}")
    print()
    print(f"Worst stratum: {s['worst_stratum']} ({s['min_stratum_accuracy']:.3f})")
    print()
    print("Per-stratum:")
    print(f"  {'stratum':32s} {'n':>3} {'correct':>7} {'acc':>6} {'avg_time':>9}")
    for name, m in sorted(s["per_stratum"].items()):
        print(f"  {name:32s} {m['n']:3d} {m['correct']:7d} {m['accuracy']:6.3f} {m['avg_time']:8.2f}s")
    print()
    print(f"Timing: avg {s['avg_time_sec']:.2f}s, p95 {s['p95_time_sec']:.2f}s")
    print("=" * 72)


# ── Driver ──────────────────────────────────────────────────────────────────

def _print_summary(cases: List[Case]) -> None:
    from collections import Counter

    print(f"Total cases: {len(cases)}")
    print(f"Solvable:   {sum(1 for c in cases if c.expected_solvable)}")
    print(f"Unsolvable: {sum(1 for c in cases if not c.expected_solvable)}")

    by_grid = Counter(c.grid_size for c in cases)
    print("\nBy grid size:")
    for g in sorted(by_grid):
        print(f"  {g}^3: {by_grid[g]}")

    print("\nBy stratum:")
    by_stratum = Counter(c.stratum for c in cases)
    for s in sorted(by_stratum):
        print(f"  {s:30s} {by_stratum[s]}")

    print("\nBy generator:")
    by_gen = Counter(c.generator for c in cases)
    for g in sorted(by_gen):
        print(f"  {g:25s} {by_gen[g]}")


if __name__ == "__main__":
    print("Building A+-target fixture (seed=561)...")
    cases = build_a_plus_fixture(seed=DEFAULT_SEED)

    print()
    _print_summary(cases)

    path = save_fixture(cases)
    print(f"\nSaved to {path}")

    # Roundtrip sanity check
    print("\nRoundtrip check: loading from disk...")
    loaded = load_fixture(path)
    assert len(loaded) == len(cases)
    for orig, load in zip(cases, loaded):
        assert orig.case_id == load.case_id
        assert orig.grid_size == load.grid_size
        assert orig.expected_solvable == load.expected_solvable
        assert len(orig.pieces) == len(load.pieces)
    print("Roundtrip OK.")
