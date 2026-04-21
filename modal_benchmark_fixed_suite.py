import json
from pathlib import Path

import modal

app = modal.App("polycube-fixed-suite-benchmark")
volume = modal.Volume.from_name("polycube-artifacts", create_if_missing=True)

VOLUME_ROOT = "/artifacts"
REPORT_DIR = "reports"

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
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


@app.function(image=image, gpu="T4", timeout=7200, volumes={VOLUME_ROOT: volume})
def run_benchmark(
    model_name: str,
    suite_path: str,
    timeout_nn: float = 24.0,
    timeout_dlx: float = 0.1,
    save_name: str = "fixed_suite_benchmark",
):
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/polycube")

    from benchmark_fixed_suite import _bucket_summary, _evaluate_case, _load_suite

    suite_candidate = Path(suite_path)
    if not suite_candidate.exists():
        mounted_candidate = Path("/root/polycube") / suite_path
        if mounted_candidate.exists():
            suite_path = str(mounted_candidate)

    suite_path_obj, suite = _load_suite(suite_path)
    grid_size = int(suite["grid_size"])
    rows = [
        _evaluate_case(
            model_name=model_name,
            case=case,
            grid_size=grid_size,
            timeout_nn=timeout_nn,
            timeout_dlx=timeout_dlx,
            device="cuda",
        )
        for case in suite["cases"]
    ]

    bucket_names = []
    for case in suite["cases"]:
        bucket = case.get("bucket")
        if bucket not in bucket_names:
            bucket_names.append(bucket)
    bucket_summaries = {
        bucket: _bucket_summary([row for row in rows if row["bucket"] == bucket])
        for bucket in bucket_names
    }
    overall = _bucket_summary(rows)

    report = {
        "model_name": model_name,
        "suite_path": str(suite_path_obj),
        "suite_name": suite.get("suite_name"),
        "grid_size": grid_size,
        "timeout_nn": timeout_nn,
        "timeout_dlx": timeout_dlx,
        "device": "cuda",
        "overall": overall,
        "bucket_summaries": bucket_summaries,
        "rows": rows,
    }

    rel_path = _volume_relpath(save_name)
    out_path = Path(VOLUME_ROOT) / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    volume.commit()

    return {
        "save_name": save_name,
        "report_relpath": rel_path,
        "overall": overall,
        "bucket_summaries": bucket_summaries,
    }


@app.local_entrypoint()
def main(
    model_name: str,
    suite_path: str = "reports/6x6_benchmark_suite_v2.json",
    timeout_nn: float = 24.0,
    timeout_dlx: float = 0.1,
    save_name: str = "6x6_fixed_suite_eval",
    fetch_only: bool = False,
):
    rel_path = _volume_relpath(save_name)
    out_path = Path("reports") / f"{save_name}.json"

    if not fetch_only:
        result = run_benchmark.remote(
            model_name=model_name,
            suite_path=suite_path,
            timeout_nn=timeout_nn,
            timeout_dlx=timeout_dlx,
            save_name=save_name,
        )
        print(
            "Remote fixed-suite benchmark saved to Modal Volume at "
            f"{result['report_relpath']}"
        )
        overall = result["overall"]
        print(
            f"overall: solve_rate={overall['solve_rate']:.3f} "
            f"valid={overall['valid_solved']}/{overall['cases']} "
            f"avg_time={overall['avg_time_sec']:.3f}s"
        )
        for bucket, summary in result["bucket_summaries"].items():
            print(
                f"{bucket}: solve_rate={summary['solve_rate']:.3f} "
                f"valid={summary['valid_solved']}/{summary['cases']} "
                f"avg_time={summary['avg_time_sec']:.3f}s"
            )

    _download_volume_file(volume, rel_path, out_path)
    print(f"Benchmark report saved to {out_path}")
