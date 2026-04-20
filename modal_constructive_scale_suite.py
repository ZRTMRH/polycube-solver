import json
from pathlib import Path

import modal

app = modal.App("polycube-constructive-scale-suite")
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
    grid_size: int,
    connected_cases: int = 20,
    robust_cases: int = 10,
    new_hard_cases: int = 10,
    seed: int = 0,
    save_name: str = "",
):
    import sys

    sys.path.insert(0, "/root/polycube")

    from build_constructive_scale_suite import build_suite as build_local
    from build_6x6_benchmark_suite import _json_default

    report = build_local(
        grid_size=int(grid_size),
        connected_cases=int(connected_cases),
        robust_cases=int(robust_cases),
        new_hard_cases=int(new_hard_cases),
        seed=int(seed),
    )
    if not save_name:
        save_name = f"{grid_size}x{grid_size}x{grid_size}_constructive_scale_suite"
    report["save_name"] = save_name

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
    grid_size: int,
    connected_cases: int = 20,
    robust_cases: int = 10,
    new_hard_cases: int = 10,
    seed: int = 0,
    save_name: str = "",
    fetch_only: bool = False,
):
    if not save_name:
        save_name = f"{grid_size}x{grid_size}x{grid_size}_constructive_scale_suite"
    rel_path = _volume_relpath(save_name)
    out_path = Path("reports") / f"{save_name}.json"

    if not fetch_only:
        result = build_suite.remote(
            grid_size=grid_size,
            connected_cases=connected_cases,
            robust_cases=robust_cases,
            new_hard_cases=new_hard_cases,
            seed=seed,
            save_name=save_name,
        )
        print(
            "Remote constructive scale suite saved to Modal Volume at "
            f"{result['report_relpath']}"
        )
        for bucket, summary in result["bucket_summaries"].items():
            print(
                f"{bucket}: cases={summary['cases']} "
                f"avg_num_pieces={summary['avg_num_pieces']:.2f} "
                f"avg_piece_size={summary['avg_avg_piece_size']:.2f} "
                f"avg_new_hard_score={summary['avg_new_hard_score']:.3f}"
            )

    _download_volume_file(volume, rel_path, out_path)
    print(f"Suite report saved to {out_path}")
