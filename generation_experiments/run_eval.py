"""
Run hybrid solver evaluation against the eval fixture.

Local:
    python generation_experiments/run_eval.py

Modal (parallel, one container per case):
    modal run generation_experiments/run_eval.py
    modal run generation_experiments/run_eval.py --model-name 4x4x4_modal --beam-width 64
"""

import json
import sys
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = "4x4x4_modal"
BEAM_WIDTH = 32
TIMEOUT_NN = 10.0
TIMEOUT_DLX = 30.0

app = modal.App("polycube-eval")
volume = modal.Volume.from_name("polycube-artifacts", create_if_missing=True)
VOLUME_ROOT = "/artifacts"

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
    .add_local_dir(
        str(ROOT), remote_path="/root/polycube",
        ignore=[".venv", "__pycache__", "*.pyc", "fixtures"],
    )
)


FIXTURE_PATH = ROOT / "fixtures" / "eval_4_to_12.json"


def _load_or_generate_fixture():
    if not FIXTURE_PATH.exists():
        raise FileNotFoundError(
            f"Fixture not found at {FIXTURE_PATH}. "
            "Generate it first: python generation_experiments/build_eval_fixture.py"
        )
    print(f"Loading existing fixture from {FIXTURE_PATH}")
    return json.loads(FIXTURE_PATH.read_text())


def _deserialize_cases(case_dicts):
    from grading_harness import Case
    return [
        Case(
            case_id=d["case_id"],
            grid_size=d["grid_size"],
            pieces=[[tuple(cell) for cell in piece] for piece in d["pieces"]],
            expected_solvable=d["expected_solvable"],
            stratum=d.get("stratum", ""),
            generator=d.get("generator", ""),
        )
        for d in case_dicts
    ]


def _score_and_print(results):
    n = len(results)
    correct = sum(1 for r in results if r.get("correct"))
    solvable = [r for r in results if r.get("expected_solvable")]
    unsolvable = [r for r in results if not r.get("expected_solvable")]

    print(f"\n{'='*60}")
    print(f"Results: {correct}/{n}  ({correct/n:.1%})")
    print(f"Solvable:   {sum(1 for r in solvable if r['correct'])}/{len(solvable)}")
    print(f"Unsolvable: {sum(1 for r in unsolvable if r['correct'])}/{len(unsolvable)}")
    print("\nBy grid size:")
    by_grid = {}
    for r in results:
        by_grid.setdefault(r["grid_size"], []).append(r)
    for g in sorted(by_grid):
        rows = by_grid[g]
        acc = sum(1 for r in rows if r["correct"]) / len(rows)
        avg_t = sum(r["time_sec"] for r in rows) / len(rows)
        print(f"  {g}^3: {acc:.0%} ({len(rows)} cases, avg {avg_t:.1f}s)")
    print(f"{'='*60}")


# ── Modal remote function ─────────────────────────────────────────────────────

@app.function(image=image, cpu=2, timeout=120, volumes={VOLUME_ROOT: volume})
def _eval_case_remote(case_dict: dict, model_name: str, beam_width: int,
                      timeout_nn: float, timeout_dlx: float) -> dict:
    import sys
    import time
    sys.path.insert(0, "/root/polycube")

    from pathlib import Path as P
    from hybrid_solver import hybrid_solve
    from grading_harness import verify_solution

    model_path = P("/root/polycube/phase2/trained_models") / f"{model_name}.pt"
    if not model_path.exists():
        vol_path = P(VOLUME_ROOT) / f"checkpoints/{model_name}.pt"
        if vol_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(vol_path.read_bytes())

    pieces = [[tuple(cell) for cell in piece] for piece in case_dict["pieces"]]
    grid_size = case_dict["grid_size"]
    expected_solvable = case_dict["expected_solvable"]

    t0 = time.time()
    try:
        raw = hybrid_solve(
            pieces,
            grid_size=grid_size,
            model_name=model_name,
            beam_width=beam_width,
            timeout_nn=timeout_nn,
            timeout_dlx=timeout_dlx,
            device="cpu",
            verbose=False,
        )
    except Exception as e:
        raw = {"solution": None, "submethod": f"solver_error:{type(e).__name__}"}
    elapsed = time.time() - t0

    solution = raw.get("solution") if isinstance(raw, dict) else None
    submethod = raw.get("submethod", "") if isinstance(raw, dict) else ""
    predicted_solvable = solution is not None

    if solution is None:
        proof_status = ("proved_unsolvable" if "no_solution" in submethod
                        else "guessed_unsolvable_on_timeout" if "timeout" in submethod
                        else "claimed_unsolvable")
    else:
        proof_status = "solution_produced"

    is_correct = False
    reason = ""
    if expected_solvable:
        if solution is None:
            reason = f"missed solvable ({proof_status})"
        elif verify_solution(solution, grid_size):
            is_correct = True
        else:
            reason = "returned invalid solution"
    else:
        is_correct = solution is None
        if not is_correct:
            reason = "returned solution for unsolvable case"

    return {
        "case_id": case_dict["case_id"],
        "grid_size": grid_size,
        "stratum": case_dict.get("stratum", ""),
        "generator": case_dict.get("generator", ""),
        "expected_solvable": expected_solvable,
        "predicted_solvable": predicted_solvable,
        "correct": is_correct,
        "time_sec": elapsed,
        "submethod": submethod,
        "proof_status": proof_status,
        "reason": reason,
    }


# ── Modal entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    model_name: str = MODEL,
    beam_width: int = BEAM_WIDTH,
    timeout_nn: float = TIMEOUT_NN,
    timeout_dlx: float = TIMEOUT_DLX,
):
    fixture = _load_or_generate_fixture()
    cases = fixture["cases"]
    print(f"Loaded {len(cases)} cases — dispatching to Modal...")

    results = list(_eval_case_remote.map(
        cases,
        kwargs=dict(model_name=model_name, beam_width=beam_width,
                    timeout_nn=timeout_nn, timeout_dlx=timeout_dlx),
        return_exceptions=True,
    ))

    clean = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"  Case {i} failed remotely: {r}")
            clean.append({"case_id": cases[i]["case_id"], "grid_size": cases[i]["grid_size"],
                          "correct": False, "expected_solvable": cases[i]["expected_solvable"],
                          "time_sec": 0, "error": str(r)})
        else:
            clean.append(r)

    _score_and_print(clean)


# ── Local entrypoint ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from fixture_a_plus import run_fixture_grading, print_fixture_report

    fixture = _load_or_generate_fixture()
    cases = _deserialize_cases(fixture["cases"])
    print(f"Loaded {len(cases)} cases\n")

    report = run_fixture_grading(
        cases, mode="hybrid", model_name=MODEL, beam_width=BEAM_WIDTH,
        timeout_nn=TIMEOUT_NN, timeout_dlx=TIMEOUT_DLX,
        timeout_per_case=TIMEOUT_DLX + TIMEOUT_NN + 5.0, verbose=True,
    )

    print_fixture_report(report)
