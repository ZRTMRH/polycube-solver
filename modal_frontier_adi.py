import io
import json
from pathlib import Path

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


@app.function(image=image, gpu="T4", timeout=7200, volumes={VOLUME_ROOT: volume})
def frontier_adi(
    base_model_name: str,
    save_name: str,
    grid_size: int = 5,
    max_pieces: int = 42,
    bootstrap_instances: int = 160,
    frontier_instances: int = 40,
    adi_epochs: int = 20,
    batch_size: int = 48,
    lr: float = 5e-4,
    lambda_value: float = 1.0,
    lambda_policy: float = 0.3,
    instance_seed: int = 42,
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

    model, _, base_metadata = load_model(base_model_name, device="cuda")

    bootstrap = generate_constructive_puzzle_instances(
        num_instances=bootstrap_instances,
        grid_size=grid_size,
        seed=instance_seed,
        large_suite_type="mixed",
        verbose=True,
    )
    existing_examples = generate_training_data(
        bootstrap,
        max_pieces=max_pieces,
        num_negatives_per_solution=0,
        verbose=True,
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
        existing_examples=existing_examples,
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
        "instance_seed": instance_seed,
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
    bootstrap_instances: int = 160,
    frontier_instances: int = 40,
    adi_epochs: int = 20,
    batch_size: int = 48,
    lr: float = 5e-4,
    lambda_value: float = 1.0,
    lambda_policy: float = 0.3,
    instance_seed: int = 42,
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
            instance_seed=instance_seed,
        )
        print(
            "Remote frontier-ADI checkpoint saved to Modal Volume at "
            f"{result['checkpoint_relpath']}"
        )

    _download_volume_file(volume, checkpoint_rel, out_path)
    _download_volume_file(volume, metadata_rel, metadata_out)
    print(f"Model saved to {out_path}")
    print(f"Metadata saved to {metadata_out}")
