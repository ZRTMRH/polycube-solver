import io
import json
from pathlib import Path
from collections import Counter

import modal

app = modal.App("polycube-frontier-adi")
volume = modal.Volume.from_name("polycube-artifacts", create_if_missing=True)

VOLUME_ROOT = "/artifacts"
CHECKPOINT_DIR = "checkpoints"
METADATA_DIR = "metadata"

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
    .add_local_dir(".", remote_path="/root/polycube", ignore=[".venv", "__pycache__", "*.pyc"])
)


def _volume_relpaths(save_name):
    return (
        f"{CHECKPOINT_DIR}/{save_name}.pt",
        f"{METADATA_DIR}/{save_name}.json",
    )


def _download_volume_file(vol, rel_path, local_path):
    chunks = list(vol.read_file(rel_path))
    if not chunks:
        raise FileNotFoundError(f"No artifact found in Modal Volume at '{rel_path}'")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(b"".join(chunks))


def _parse_variant_spec(spec: str):
    """Parse comma-separated variant names with optional integer weights.

    Examples:
        "connected,robust"
        "connected*2,robust"
    """
    entries = []
    for raw in str(spec).split(","):
        token = raw.strip()
        if not token:
            continue
        weight = 1
        if "*" in token:
            name, weight_str = token.split("*", 1)
            token = name.strip()
            weight = max(1, int(weight_str))
        entries.append((token, weight))
    if not entries:
        raise ValueError(f"Empty variant spec: {spec!r}")
    return entries


def _allocate_counts(total, weighted_variants):
    total = int(total)
    weighted_variants = list(weighted_variants)
    if total <= 0:
        return []
    total_weight = sum(weight for _, weight in weighted_variants)
    counts = []
    remaining = total
    for idx, (name, weight) in enumerate(weighted_variants):
        if idx == len(weighted_variants) - 1:
            count = remaining
        else:
            count = max(0, round(total * weight / total_weight))
            count = min(count, remaining)
        counts.append((name, count))
        remaining -= count
    if remaining > 0 and counts:
        name, count = counts[-1]
        counts[-1] = (name, count + remaining)
    return [(name, count) for name, count in counts if count > 0]


def _canonical_piece(piece):
    from phase1.polycube import normalize

    return tuple(sorted(tuple(cell) for cell in normalize(piece)))


def _canonical_case_key(grid_size, pieces):
    return (
        int(grid_size),
        tuple(sorted(_canonical_piece(piece) for piece in pieces)),
    )


def _difficulty_metrics(grid_size, pieces):
    from phase1.polycube import get_all_placements

    sizes = [len(piece) for piece in pieces]
    placements = [len(get_all_placements(piece, grid_size)) for piece in pieces]
    shape_counts = Counter(_canonical_piece(piece) for piece in pieces)
    repeated_shape_fraction = (
        sum(v for v in shape_counts.values() if v > 1) / max(1, len(pieces))
    )

    metrics = {
        "num_pieces": len(pieces),
        "avg_piece_size": sum(sizes) / max(1, len(sizes)),
        "avg_piece_placements": sum(placements) / max(1, len(placements)),
        "max_piece_placements": max(placements) if placements else 0,
        "repeated_shape_fraction": repeated_shape_fraction,
        "unique_shapes": len(shape_counts),
    }
    metrics["legacy_style_score"] = (
        3.4 * metrics["avg_piece_size"]
        + 4.0 * metrics["repeated_shape_fraction"]
        - 0.08 * metrics["num_pieces"]
        + 0.0004 * metrics["avg_piece_placements"]
    )
    metrics["new_hard_score"] = (
        2.5 * metrics["avg_piece_size"]
        + 3.0 * metrics["repeated_shape_fraction"]
        - 0.05 * metrics["num_pieces"]
        + 0.0010 * metrics["avg_piece_placements"]
    )
    return metrics


def _rank_instances_by_score(instances, score_key, grid_size, limit):
    ranked = []
    seen = set()
    for inst in instances:
        key = _canonical_case_key(grid_size, inst["pieces"])
        if key in seen:
            continue
        seen.add(key)
        ranked.append((inst, _difficulty_metrics(grid_size, inst["pieces"])))

    ranked.sort(
        key=lambda pair: (
            pair[1][score_key],
            pair[1]["avg_piece_size"],
            pair[1]["repeated_shape_fraction"],
            -pair[1]["num_pieces"],
            pair[1]["avg_piece_placements"],
        ),
        reverse=True,
    )

    selected = []
    for inst, metrics in ranked[:limit]:
        row = dict(inst)
        row["selection_metrics"] = metrics
        selected.append(row)
    return selected


def _build_direct_constructive_pool(
    *,
    generate_constructive_puzzle_instances,
    num_instances,
    grid_size,
    variant_spec,
    seed,
    verbose,
):
    """Generate a mixed constructive pool from direct generator families."""
    weighted_variants = _parse_variant_spec(variant_spec)
    allocations = _allocate_counts(num_instances, weighted_variants)

    all_instances = []
    instance_source_counts = Counter()
    variant_request_counts = {}
    seed_step = 10000

    for idx, (variant_name, count) in enumerate(allocations):
        variant_request_counts[variant_name] = count
        variant_seed = int(seed) + idx * seed_step
        if verbose:
            print(
                f"  Generating {count} instances from constructive variant "
                f"{variant_name} (seed={variant_seed})..."
            )
        instances = generate_constructive_puzzle_instances(
            num_instances=count,
            grid_size=grid_size,
            seed=variant_seed,
            large_suite_type=variant_name,
            verbose=verbose,
        )
        for inst in instances:
            inst["variant_request"] = variant_name
            instance_source_counts[inst.get("instance_source")] += 1
        all_instances.extend(instances)

    return all_instances, {
        "variant_spec": variant_spec,
        "variant_request_counts": variant_request_counts,
        "instance_source_counts": dict(instance_source_counts),
    }


def _build_special_constructive_pool(
    *,
    generate_constructive_puzzle_instances,
    num_instances,
    grid_size,
    variant_name,
    seed,
    verbose,
):
    if variant_name in {"legacy", "legacy_hard"}:
        score_key = "legacy_style_score"
        candidate_variant_spec = "connected,robust"
    elif variant_name in {"new_hard", "newhard", "connected_hard"}:
        score_key = "new_hard_score"
        candidate_variant_spec = "connected"
    else:
        raise ValueError(f"Unsupported special constructive variant '{variant_name}'.")

    oversample = max(num_instances * 6, num_instances * 2)
    candidate_instances, candidate_meta = _build_direct_constructive_pool(
        generate_constructive_puzzle_instances=generate_constructive_puzzle_instances,
        num_instances=oversample,
        grid_size=grid_size,
        variant_spec=candidate_variant_spec,
        seed=seed,
        verbose=verbose,
    )
    selected = _rank_instances_by_score(
        candidate_instances,
        score_key=score_key,
        grid_size=grid_size,
        limit=num_instances,
    )
    source_counts = Counter()
    for inst in selected:
        inst["variant_request"] = variant_name
        source_counts[inst.get("instance_source")] += 1
    return selected, {
        "variant_name": variant_name,
        "score_key": score_key,
        "candidate_variant_spec": candidate_variant_spec,
        "candidate_meta": candidate_meta,
        "selected_instance_source_counts": dict(source_counts),
    }


def _build_constructive_pool(
    *,
    generate_constructive_puzzle_instances,
    num_instances,
    grid_size,
    variant_spec,
    seed,
    verbose,
):
    """Generate a mixed constructive pool from one or more families."""
    weighted_variants = _parse_variant_spec(variant_spec)
    allocations = _allocate_counts(num_instances, weighted_variants)

    all_instances = []
    instance_source_counts = Counter()
    variant_request_counts = {}
    generation_details = {}
    seed_step = 10000

    for idx, (variant_name, count) in enumerate(allocations):
        variant_request_counts[variant_name] = count
        variant_seed = int(seed) + idx * seed_step
        if variant_name in {"legacy", "legacy_hard", "new_hard", "newhard", "connected_hard"}:
            instances, detail = _build_special_constructive_pool(
                generate_constructive_puzzle_instances=generate_constructive_puzzle_instances,
                num_instances=count,
                grid_size=grid_size,
                variant_name=variant_name,
                seed=variant_seed,
                verbose=verbose,
            )
            generation_details[variant_name] = detail
        else:
            instances, detail = _build_direct_constructive_pool(
                generate_constructive_puzzle_instances=generate_constructive_puzzle_instances,
                num_instances=count,
                grid_size=grid_size,
                variant_spec=variant_name,
                seed=variant_seed,
                verbose=verbose,
            )
            generation_details[variant_name] = detail

        for inst in instances:
            instance_source_counts[inst.get("instance_source")] += 1
        all_instances.extend(instances)

    return all_instances, {
        "variant_spec": variant_spec,
        "variant_request_counts": variant_request_counts,
        "instance_source_counts": dict(instance_source_counts),
        "variant_generation_details": generation_details,
    }


def _load_proxy_cases(suite_path, cases_per_bucket):
    if not suite_path:
        return []
    suite_file = Path(suite_path)
    if not suite_file.exists():
        alt = Path("/root/polycube") / suite_path
        if alt.exists():
            suite_file = alt
    data = json.loads(suite_file.read_text())
    grouped = {}
    bucket_order = []
    for case in data.get("cases", []):
        bucket = case.get("bucket")
        if bucket not in grouped:
            grouped[bucket] = []
            bucket_order.append(bucket)
        grouped[bucket].append(case)

    selected = []
    per_bucket = max(0, int(cases_per_bucket))
    if per_bucket <= 0:
        return selected
    for bucket in bucket_order:
        selected.extend(grouped.get(bucket, [])[:per_bucket])
    return selected


def _make_proxy_eval_callback(proxy_cases, grid_size, timeout_nn, timeout_dlx, device):
    if not proxy_cases:
        return None

    from collections import defaultdict
    from hybrid_solver import hybrid_solve
    from phase1.test_cases import verify_solution

    def _callback(*, model, epochs_trained):
        rows = []
        by_bucket = defaultdict(list)
        for case in proxy_cases:
            result = hybrid_solve(
                case["pieces"],
                grid_size=grid_size,
                model=model,
                model_name=None,
                timeout_nn=timeout_nn,
                timeout_dlx=timeout_dlx,
                device=device,
                verbose=False,
            )
            solution = result.get("solution")
            solved = solution is not None
            valid = verify_solution(solution, grid_size, case["pieces"]) if solved else False
            row = {
                "case_id": case["case_id"],
                "bucket": case.get("bucket"),
                "valid": valid,
                "submethod": result.get("submethod"),
            }
            rows.append(row)
            by_bucket[row["bucket"]].append(row)

        bucket_summaries = {}
        worst_bucket = 1.0
        for bucket, bucket_rows in by_bucket.items():
            solved = sum(1 for row in bucket_rows if row["valid"])
            solve_rate = solved / max(1, len(bucket_rows))
            bucket_summaries[bucket] = {
                "cases": len(bucket_rows),
                "valid_solved": solved,
                "solve_rate": solve_rate,
            }
            worst_bucket = min(worst_bucket, solve_rate)

        overall_solved = sum(1 for row in rows if row["valid"])
        overall_rate = overall_solved / max(1, len(rows))
        score = overall_rate + 0.5 * worst_bucket
        return {
            "overall_solve_rate": overall_rate,
            "worst_bucket_solve_rate": worst_bucket,
            "bucket_summaries": bucket_summaries,
            "score": score,
            "cases": len(rows),
        }

    return _callback


@app.function(image=image, gpu="T4", timeout=14400, volumes={VOLUME_ROOT: volume})
def frontier_adi(
    base_model_name: str,
    save_name: str,
    grid_size: int = 5,
    max_pieces: int = 42,
    bootstrap_instances: int = 240,
    frontier_instances: int = 120,
    adi_epochs: int = 6,
    batch_size: int = 32,
    lr: float = 1.5e-4,
    lambda_value: float = 1.0,
    lambda_policy: float = 0.35,
    bootstrap_variant: str = "connected,robust,new_hard",
    frontier_variant: str = "robust*2,new_hard*2",
    instance_seed: int = 42,
    proxy_suite_path: str = "",
    proxy_cases_per_bucket: int = 0,
    proxy_eval_every: int = 0,
    proxy_patience: int = 0,
):
    import sys
    import torch

    sys.path.insert(0, "/root/polycube")

    from phase2.train import (
        load_model,
        run_frontier_adi_iteration,
    )
    from phase2.data_generator import (
        generate_constructive_puzzle_instances,
        generate_training_data,
    )

    grid_size = int(grid_size)
    max_pieces = int(max_pieces)
    bootstrap_instances = int(bootstrap_instances)
    frontier_instances = int(frontier_instances)
    adi_epochs = int(adi_epochs)
    batch_size = int(batch_size)
    lr = float(lr)
    lambda_value = float(lambda_value)
    lambda_policy = float(lambda_policy)
    instance_seed = int(instance_seed)
    proxy_cases_per_bucket = int(proxy_cases_per_bucket)
    proxy_eval_every = int(proxy_eval_every)
    proxy_patience = int(proxy_patience)

    model, _, base_metadata = load_model(
        base_model_name,
        device="cuda",
        grid_size_override=grid_size,
        max_pieces_override=max_pieces,
    )

    bootstrap, bootstrap_mix = _build_constructive_pool(
        generate_constructive_puzzle_instances=generate_constructive_puzzle_instances,
        num_instances=bootstrap_instances,
        grid_size=grid_size,
        variant_spec=bootstrap_variant,
        seed=instance_seed,
        verbose=True,
    )
    existing_examples = generate_training_data(
        bootstrap,
        max_pieces=max_pieces,
        num_negatives_per_solution=0,
        verbose=True,
    )

    frontier_instances_pool, frontier_mix = _build_constructive_pool(
        generate_constructive_puzzle_instances=generate_constructive_puzzle_instances,
        num_instances=frontier_instances,
        grid_size=grid_size,
        variant_spec=frontier_variant,
        seed=instance_seed + 1000,
        verbose=True,
    )

    proxy_cases = _load_proxy_cases(proxy_suite_path, proxy_cases_per_bucket)
    proxy_callback = _make_proxy_eval_callback(
        proxy_cases,
        grid_size=grid_size,
        timeout_nn=24.0,
        timeout_dlx=0.1,
        device="cuda",
    )

    model, adi_history, frontier_examples = run_frontier_adi_iteration(
        model=model,
        grid_size=grid_size,
        max_pieces=max_pieces,
        num_new_instances=frontier_instances,
        adi_epochs=adi_epochs,
        lr=lr,
        batch_size=batch_size,
        lambda_value=lambda_value,
        lambda_policy=lambda_policy,
        search_timeout=8.0,
        frontier_verify_roots=6,
        frontier_verify_timeout=18.0,
        frontier_verify_branch_limit=6,
        frontier_verify_max_nodes=25000,
        frontier_verify_ranker="contact",
        instance_source="constructive",
        constructive_variant="mixed",
        instance_seed=instance_seed + 1000,
        adi_instances=frontier_instances_pool,
        existing_examples=existing_examples,
        eval_callback=proxy_callback,
        eval_every_epochs=proxy_eval_every,
        early_stop_patience=proxy_patience,
        device="cuda",
        verbose=True,
    )

    metadata = {
        "grid_size": grid_size,
        "max_pieces": max_pieces,
        "base_model_name": base_model_name,
        "bootstrap_instances": bootstrap_instances,
        "frontier_instances": frontier_instances,
        "frontier_examples": len(frontier_examples),
        "adi_epochs": adi_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lambda_value": lambda_value,
        "lambda_policy": lambda_policy,
        "bootstrap_variant": bootstrap_variant,
        "frontier_variant": frontier_variant,
        "bootstrap_mix": bootstrap_mix,
        "frontier_mix": frontier_mix,
        "instance_seed": instance_seed,
        "proxy_suite_path": proxy_suite_path,
        "proxy_cases_per_bucket": proxy_cases_per_bucket,
        "proxy_eval_every": proxy_eval_every,
        "proxy_patience": proxy_patience,
        "proxy_eval_history": adi_history.get("proxy_eval", []),
        "epochs_completed": adi_history.get("_epochs_completed", adi_epochs),
        "best_proxy_score": adi_history.get("_best_proxy_score"),
        "base_metadata": base_metadata,
        "final_val_acc": adi_history["val_value_acc"][-1] if adi_history.get("val_value_acc") else None,
    }

    buf = io.BytesIO()
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_channels": model.in_channels,
        "grid_size": model.grid_size,
        "hidden_dim": model.hidden_dim,
        "num_residual_blocks": len(model.body),
        "value_head_type": getattr(model, "value_head_type", "fc"),
        "policy_head_type": getattr(model, "policy_head_type", "fc"),
        "use_context_features": getattr(model, "use_context_features", False),
        "history": adi_history,
        "metadata": metadata,
    }, buf)

    checkpoint_rel, metadata_rel = _volume_relpaths(save_name)
    checkpoint_path = Path(VOLUME_ROOT) / checkpoint_rel
    metadata_path = Path(VOLUME_ROOT) / metadata_rel
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(buf.getvalue())
    metadata_path.write_text(json.dumps(metadata, indent=2))
    volume.commit()

    return {
        "save_name": save_name,
        "checkpoint_relpath": checkpoint_rel,
        "metadata_relpath": metadata_rel,
        "metadata": metadata,
    }


@app.local_entrypoint()
def main(
    base_model_name: str,
    save_name: str,
    grid_size: int = 5,
    max_pieces: int = 42,
    bootstrap_instances: int = 240,
    frontier_instances: int = 120,
    adi_epochs: int = 6,
    batch_size: int = 32,
    lr: float = 1.5e-4,
    lambda_value: float = 1.0,
    lambda_policy: float = 0.35,
    bootstrap_variant: str = "connected,robust,new_hard",
    frontier_variant: str = "robust*2,new_hard*2",
    instance_seed: int = 42,
    proxy_suite_path: str = "",
    proxy_cases_per_bucket: int = 0,
    proxy_eval_every: int = 0,
    proxy_patience: int = 0,
    fetch_only: bool = False,
):
    checkpoint_rel, metadata_rel = _volume_relpaths(save_name)
    out_path = Path("phase2/trained_models") / f"{save_name}.pt"
    metadata_out = Path("phase2/trained_models") / f"{save_name}.metadata.json"

    if not fetch_only:
        result = frontier_adi.remote(
            base_model_name=base_model_name,
            save_name=save_name,
            grid_size=grid_size,
            max_pieces=max_pieces,
            bootstrap_instances=bootstrap_instances,
            frontier_instances=frontier_instances,
            adi_epochs=adi_epochs,
            batch_size=batch_size,
            lr=lr,
            lambda_value=lambda_value,
            lambda_policy=lambda_policy,
            bootstrap_variant=bootstrap_variant,
            frontier_variant=frontier_variant,
            instance_seed=instance_seed,
            proxy_suite_path=proxy_suite_path,
            proxy_cases_per_bucket=proxy_cases_per_bucket,
            proxy_eval_every=proxy_eval_every,
            proxy_patience=proxy_patience,
        )
        print(
            "Remote frontier-ADI checkpoint saved to Modal Volume at "
            f"{result['checkpoint_relpath']}"
        )

    _download_volume_file(volume, checkpoint_rel, out_path)
    _download_volume_file(volume, metadata_rel, metadata_out)
    print(f"Model saved to {out_path}")
    print(f"Metadata saved to {metadata_out}")
