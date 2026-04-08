import modal

app = modal.App("polycube-trainer")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
    .add_local_dir(".", remote_path="/root/polycube", ignore=[".venv", "__pycache__", "*.pyc"])
)


@app.function(image=image, gpu="T4", timeout=3600)
def train(grid_size=4, max_pieces=21, num_instances=50, epochs=50):
    import sys
    import io
    import torch
    sys.path.insert(0, "/root/polycube")

    from phase2.train import run_supervised_training

    model, history, _ = run_supervised_training(
        grid_size=grid_size,
        max_pieces=max_pieces,
        num_instances=num_instances,
        epochs=epochs,
        device="cuda",
    )

    # Serialize checkpoint and return as bytes so it can be saved locally
    buf = io.BytesIO()
    torch.save({
        'model_state_dict': model.state_dict(),
        'in_channels': model.in_channels,
        'grid_size': model.grid_size,
        'hidden_dim': model.hidden_dim,
        'num_residual_blocks': len(model.body),
        'history': history,
    }, buf)
    return buf.getvalue()


@app.local_entrypoint()
def main():
    from pathlib import Path

    checkpoint_bytes = train.remote()

    out_path = Path("phase2/trained_models/4x4x4_modal.pt")
    out_path.write_bytes(checkpoint_bytes)
    print(f"Model saved to {out_path}")
