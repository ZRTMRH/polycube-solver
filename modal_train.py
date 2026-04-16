import json
from pathlib import Path

import modal

app = modal.App("polycube-trainer")
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


def _resolve_recipe_overrides(grid_size, **overrides):
    from phase2.training_recipes import recommended_training_recipe

    recipe = recommended_training_recipe(grid_size)
    for key, value in overrides.items():
        if value is not None:
            recipe[key] = value
    return recipe


@app.function(image=image, gpu="T4", timeout=3600, volumes={VOLUME_ROOT: volume})
def train(grid_size: int = 4, max_pieces: int = 21, num_instances: int | None = None,
          num_negatives: int | None = None,
          epochs: int | None = None, batch_size: int | None = None, lr: float | None = None,
          hidden_dim: int | None = None,
          num_residual_blocks: int | None = None,
          value_head_type: str | None = None, policy_head_type: str | None = None,
          use_context_features: bool | None = None,
          lambda_value: float | None = None, lambda_policy: float | None = None,
          instance_source: str | None = None,
          constructive_variant: str | None = None,
          instance_seed: int = 42,
          save_name: str = "4x4x4_modal_constructive"):
    import sys
    import io
    from pathlib import Path
    import torch
    sys.path.insert(0, "/root/polycube")

    from phase2.train import run_supervised_training

    grid_size = int(grid_size)
    max_pieces = int(max_pieces)
    if num_negatives is not None:
        num_negatives = int(num_negatives)
    if num_instances is not None:
        num_instances = int(num_instances)
    if epochs is not None:
        epochs = int(epochs)
    if batch_size is not None:
        batch_size = int(batch_size)
    if lr is not None:
        lr = float(lr)
    if lambda_value is not None:
        lambda_value = float(lambda_value)
    if lambda_policy is not None:
        lambda_policy = float(lambda_policy)
    instance_seed = int(instance_seed)

    recipe = _resolve_recipe_overrides(
        grid_size,
        num_instances=num_instances,
        num_negatives=num_negatives,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        num_residual_blocks=num_residual_blocks,
        value_head_type=value_head_type,
        policy_head_type=policy_head_type,
        use_context_features=use_context_features,
        lambda_value=lambda_value,
        lambda_policy=lambda_policy,
        instance_source=instance_source,
        constructive_variant=constructive_variant,
    )

    model, history, examples = run_supervised_training(
        grid_size=grid_size,
        max_pieces=max_pieces,
        num_instances=recipe["num_instances"],
        num_negatives=recipe["num_negatives"],
        epochs=recipe["epochs"],
        batch_size=recipe["batch_size"],
        lr=recipe["lr"],
        hidden_dim=recipe["hidden_dim"],
        num_residual_blocks=recipe["num_residual_blocks"],
        value_head_type=recipe["value_head_type"],
        policy_head_type=recipe["policy_head_type"],
        use_context_features=recipe["use_context_features"],
        lambda_value=recipe["lambda_value"],
        lambda_policy=recipe["lambda_policy"],
        instance_source=recipe["instance_source"],
        constructive_variant=recipe["constructive_variant"],
        instance_seed=instance_seed,
        device="cuda",
    )

    metadata = {
        'grid_size': grid_size,
        'max_pieces': max_pieces,
        'num_instances': recipe["num_instances"],
        'num_negatives': recipe["num_negatives"],
        'num_examples': len(examples),
        'epochs': recipe["epochs"],
        'batch_size': recipe["batch_size"],
        'lr': recipe["lr"],
        'hidden_dim': recipe["hidden_dim"],
        'num_residual_blocks': recipe["num_residual_blocks"],
        'use_context_features': recipe["use_context_features"],
        'lambda_value': recipe["lambda_value"],
        'lambda_policy': recipe["lambda_policy"],
        'instance_source': recipe["instance_source"],
        'constructive_variant': recipe["constructive_variant"],
        'instance_seed': instance_seed,
        'value_head_type': recipe["value_head_type"],
        'policy_head_type': recipe["policy_head_type"],
        'final_val_acc': history['val_value_acc'][-1] if history.get('val_value_acc') else None,
    }

    # Serialize checkpoint into the shared Modal Volume so it survives disconnects.
    buf = io.BytesIO()
    torch.save({
        'model_state_dict': model.state_dict(),
        'in_channels': model.in_channels,
        'grid_size': model.grid_size,
        'hidden_dim': model.hidden_dim,
        'num_residual_blocks': len(model.body),
        'value_head_type': getattr(model, 'value_head_type', 'fc'),
        'policy_head_type': getattr(model, 'policy_head_type', 'fc'),
        'use_context_features': getattr(model, 'use_context_features', False),
        'history': history,
        'metadata': metadata,
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
def main(grid_size: int = 4, max_pieces: int = 21, num_instances: int | None = None,
         num_negatives: int | None = None,
         hidden_dim: int | None = None,
         num_residual_blocks: int | None = None,
         use_context_features: bool | None = None,
         epochs: int | None = None, save_name: str = "4x4x4_modal_constructive",
         fetch_only: bool = False):
    checkpoint_rel, metadata_rel = _volume_relpaths(save_name)
    out_path = Path("phase2/trained_models") / f"{save_name}.pt"
    metadata_out = Path("phase2/trained_models") / f"{save_name}.metadata.json"

    if not fetch_only:
        result = train.remote(
            grid_size=grid_size,
            max_pieces=max_pieces,
            num_instances=num_instances,
            num_negatives=num_negatives,
            hidden_dim=hidden_dim,
            num_residual_blocks=num_residual_blocks,
            use_context_features=use_context_features,
            epochs=epochs,
            save_name=save_name,
        )
        print(
            "Remote checkpoint saved to Modal Volume at "
            f"{result['checkpoint_relpath']}"
        )

    _download_volume_file(volume, checkpoint_rel, out_path)
    _download_volume_file(volume, metadata_rel, metadata_out)
    print(f"Model saved to {out_path}")
    print(f"Metadata saved to {metadata_out}")
