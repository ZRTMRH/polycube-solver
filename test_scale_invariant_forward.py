"""Dry-run sanity check: scale-invariant CuboidNet forward pass at multiple N.

Instantiate a model with gap/conv heads + context features, then run forward
passes at N=5 and N=12 to confirm no shape errors and that policy output
size scales as N**3.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from phase2.nn_model import create_model, count_parameters


def main():
    max_pieces = 21  # matches modal_train.py default

    # Build a scale-invariant model. grid_size=5 here is only used for
    # bookkeeping; gap/conv heads have no grid-size-bound weights.
    model = create_model(
        grid_size=5,
        max_pieces=max_pieces,
        num_residual_blocks=8,
        hidden_dim=160,
        value_head_type='gap',
        policy_head_type='conv',
        use_context_features=True,
    )
    model.eval()

    total, trainable = count_parameters(model)
    print(f"Scale-invariant CuboidNet (gap/conv + context)")
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")
    print(f"  in_channels: {model.in_channels}, hidden_dim: {model.hidden_dim}")

    in_channels = 1 + max_pieces

    for N in (5, 7, 9, 12):
        x = torch.randn(2, in_channels, N, N, N)
        with torch.no_grad():
            value, policy = model(x)
        expected_policy = N * N * N
        assert value.shape == (2, 1), f"value shape {value.shape} != (2, 1) at N={N}"
        assert policy.shape == (2, expected_policy), (
            f"policy shape {policy.shape} != (2, {expected_policy}) at N={N}"
        )
        print(f"  N={N:2d}: value={tuple(value.shape)}, policy={tuple(policy.shape)} OK")

    # Round-trip via save/load to confirm head-type metadata persists.
    import io
    buf = io.BytesIO()
    torch.save({
        'model_state_dict': model.state_dict(),
        'in_channels': model.in_channels,
        'grid_size': model.grid_size,
        'hidden_dim': model.hidden_dim,
        'num_residual_blocks': len(model.body),
        'value_head_type': model.value_head_type,
        'policy_head_type': model.policy_head_type,
        'use_context_features': model.use_context_features,
    }, buf)
    buf.seek(0)
    blob = torch.load(buf, weights_only=False)
    restored = create_model(
        grid_size=blob['grid_size'],
        max_pieces=blob['in_channels'] - 1,
        num_residual_blocks=blob['num_residual_blocks'],
        hidden_dim=blob['hidden_dim'],
        value_head_type=blob['value_head_type'],
        policy_head_type=blob['policy_head_type'],
        use_context_features=blob['use_context_features'],
    )
    restored.load_state_dict(blob['model_state_dict'])
    restored.eval()

    x12 = torch.randn(1, in_channels, 12, 12, 12)
    with torch.no_grad():
        v_a, p_a = model(x12)
        v_b, p_b = restored(x12)
    assert torch.allclose(v_a, v_b, atol=1e-6) and torch.allclose(p_a, p_b, atol=1e-6), \
        "save/load round-trip changed outputs"
    print("  save/load round-trip preserves outputs at N=12 OK")
    print("\nAll scale-invariant forward checks passed.")


if __name__ == "__main__":
    main()
