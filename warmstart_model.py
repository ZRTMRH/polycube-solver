"""Create a resized warm-start checkpoint from an existing model.

Example:
    .venv/bin/python warmstart_model.py \
        --source-name 5x5x5_calibrated_v1_light \
        --target-name 6x6x6_warmstart_from_5x5 \
        --grid-size 6 \
        --max-pieces 64
"""

from __future__ import annotations

import argparse

from phase2.train import load_model, save_model


def main():
    parser = argparse.ArgumentParser(
        description="Resize a checkpoint into a warm-start model for a new grid size."
    )
    parser.add_argument("--source-name", required=True, type=str)
    parser.add_argument("--target-name", required=True, type=str)
    parser.add_argument("--grid-size", required=True, type=int)
    parser.add_argument("--max-pieces", required=True, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()

    model, history, metadata = load_model(
        args.source_name,
        device=args.device,
        grid_size_override=args.grid_size,
        max_pieces_override=args.max_pieces,
    )

    metadata = dict(metadata or {})
    metadata["warm_started_from"] = args.source_name
    metadata["grid_size"] = args.grid_size
    metadata["max_pieces"] = args.max_pieces

    save_model(model, args.target_name, history=history, metadata=metadata)
    print(
        "Warm-start checkpoint created: "
        f"{args.target_name} (grid={args.grid_size}, max_pieces={args.max_pieces})"
    )


if __name__ == "__main__":
    main()
