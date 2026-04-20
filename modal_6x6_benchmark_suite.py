import json
from pathlib import Path

import modal

app = modal.App("polycube-6x6-benchmark-suite")
volume = modal.Volume.from_name("polycube-artifacts", create_if_missing=True)

VOLUME_ROOT = "/artifacts"
REPORT_DIR = "reports"

image = (
    modal.Image.debian_slim()
    .pip_install("numpy")
    .add_local_dir(
        ".",
        remote_path="/root/polycube",
        ignore=[".venv", "__pycache__", "*.pyc"],
    )
)


def _volume_relpath(save_name: str) -> str:
    return f"{REPORT_DIR}/{save_name}.json"


def _download_volume_file(vol, rel_path, local_path):
    chunks = list(vol.read_file(rel_path))
    if not chunks:
        raise FileNotFoundError(f"No artifact found in Modal Volume at '{rel_path}'")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(b"".join(chunks))


@app.function(image=image, timeout=3600, volumes={VOLUME_ROOT: volume})
def build_suite(
    grid_size: int = 6,
    connected_cases: int = 20,
    legacy_cases: int = 10,
    new_hard_cases: int = 10,
    seed: int = 0,
    save_name: str = "6x6_benchmark_suite_v1",
):
    import sys

    sys.path.insert(0, "/root/polycube")

    from build_6x6_benchmark_suite import (
        _bucket_summary,
        _build_connected_cases,
        _build_hard_candidate_pool,
        _select_scored_cases,
        _json_default,
    )

    grid_size = int(grid_size)
    connected_cases = int(connected_cases)
    legacy_cases = int(legacy_cases)
    new_hard_cases = int(new_hard_cases)
    seed = int(seed)

    connected_rows = _build_connected_cases(
        grid_size=grid_size,
        n_cases=connected_cases,
        seed=seed,
    )
    candidate_pool = _build_hard_candidate_pool(
        grid_size=grid_size,
        n_cases=max(legacy_cases, new_hard_cases),
        seed=seed + 10000,
    )
    connected_candidates = [
        row for row in candidate_pool
        if row["generator_family"] == "constructive:connected"
    ]
    used_keys = set()
    legacy_rows = _select_scored_cases(
        candidate_pool,
        score_key="legacy_style_score",
        n_cases=legacy_cases,
        bucket_name="legacy_style_hard",
        grid_size=grid_size,
        used_keys=used_keys,
    )
    new_hard_rows = _select_scored_cases(
        connected_candidates,
        score_key="new_hard_score",
        n_cases=new_hard_cases,
        bucket_name="new_hard",
        grid_size=grid_size,
        used_keys=used_keys,
    )

    report = {
        "grid_size": grid_size,
        "suite_name": "representative_constructive_6x6",
        "save_name": save_name,
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
        "cases": connected_rows + legacy_rows + new_hard_rows,
    }

    rel_path = _volume_relpath(save_name)
    out_path = Path(VOLUME_ROOT) / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=_json_default))
    volume.commit()

    return {
        "save_name": save_name,
        "report_relpath": rel_path,
        "bucket_summaries": report["bucket_summaries"],
        "n_cases": len(report["cases"]),
    }


@app.local_entrypoint()
def main(
    grid_size: int = 6,
    connected_cases: int = 20,
    legacy_cases: int = 10,
    new_hard_cases: int = 10,
    seed: int = 0,
    save_name: str = "6x6_benchmark_suite_v1",
    fetch_only: bool = False,
):
    rel_path = _volume_relpath(save_name)
    out_path = Path("reports") / f"{save_name}.json"

    if not fetch_only:
        result = build_suite.remote(
            grid_size=grid_size,
            connected_cases=connected_cases,
            legacy_cases=legacy_cases,
            new_hard_cases=new_hard_cases,
            seed=seed,
            save_name=save_name,
        )
        print(
            "Remote 6x6 benchmark suite saved to Modal Volume at "
            f"{result['report_relpath']}"
        )
        for bucket, summary in result["bucket_summaries"].items():
            print(
                f"{bucket}: cases={summary['cases']} "
                f"avg_num_pieces={summary['avg_num_pieces']:.2f} "
                f"avg_piece_size={summary['avg_avg_piece_size']:.2f} "
                f"avg_legacy_style_score={summary['avg_legacy_style_score']:.3f} "
                f"avg_new_hard_score={summary['avg_new_hard_score']:.3f}"
            )

    _download_volume_file(volume, rel_path, out_path)
    print(f"Suite report saved to {out_path}")
