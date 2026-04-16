import json
from pathlib import Path

import modal

app = modal.App("polycube-profile-benchmark")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
    .add_local_dir(".", remote_path="/root/polycube", ignore=[".venv", "__pycache__", "*.pyc"])
)


@app.function(image=image, gpu="T4", timeout=1800)
def benchmark(model_name="4x4x4_adi2", grid_size=4, eval_cases=3,
              seed=1403, timeout=5.0,
              profiles=("default_like", "narrow_4x4", "balanced_4x4")):
    import sys
    import time
    sys.path.insert(0, "/root/polycube")

    from grading_harness import build_scale_suite
    from phase1.test_cases import verify_solution
    from phase2.nn_solver import nn_solve
    from phase2.search_profiles import resolve_search_profile
    from phase2.train import load_model

    device = "cuda"
    model, _, metadata = load_model(model_name, device=device)
    max_pieces = model.in_channels - 1
    cases, suite_source = build_scale_suite(
        grid_size=grid_size,
        n_cases=eval_cases,
        seed=seed,
        dlx_timeout=12.0,
    )

    report = {
        "model_name": model_name,
        "grid_size": grid_size,
        "device": device,
        "timeout": timeout,
        "suite_source": suite_source,
        "eval_cases": len(cases),
        "case_ids": [case.case_id for case in cases],
        "model_metadata": metadata,
        "profiles": [],
    }

    for profile_name in profiles:
        profile = resolve_search_profile(profile_name)
        rows = []
        for case in cases:
            t0 = time.time()
            solution, diagnostics = nn_solve(
                case.pieces,
                case.grid_size,
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
                return_diagnostics=True,
            )
            elapsed = time.time() - t0
            solved = solution is not None
            valid = verify_solution(solution, case.grid_size) if solved else False
            rows.append({
                "case_id": case.case_id,
                "solved": solved,
                "valid": valid,
                "time_sec": elapsed,
                "failure_reason": diagnostics.get("failure_reason"),
                "depth_reached": diagnostics.get("depth_reached"),
                "generated_children": diagnostics.get("generated_children"),
                "placements_dropped_total": diagnostics.get("placements_dropped_total"),
            })

        valid_solved = sum(1 for row in rows if row["valid"])
        report["profiles"].append({
            "profile_name": profile_name,
            "profile": profile,
            "n_cases": len(rows),
            "valid_solved": valid_solved,
            "solve_rate": valid_solved / max(1, len(rows)),
            "avg_time_sec": sum(row["time_sec"] for row in rows) / max(1, len(rows)),
            "rows": rows,
        })

    return report


@app.local_entrypoint()
def main(model_name: str = "4x4x4_adi2", grid_size: int = 4, eval_cases: int = 3,
         seed: int = 1403, timeout: float = 5.0,
         out: str = "reports/modal_profile_benchmark.json"):
    report = benchmark.remote(
        model_name=model_name,
        grid_size=grid_size,
        eval_cases=eval_cases,
        seed=seed,
        timeout=timeout,
    )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved report to {out_path}")

    for profile in report["profiles"]:
        print(
            f"profile={profile['profile_name']} solve_rate={profile['solve_rate']:.3f} "
            f"valid={profile['valid_solved']}/{profile['n_cases']} "
            f"avg_time={profile['avg_time_sec']:.3f}s"
        )
