"""
Training pipeline for CuboidNet.

Phase A: Supervised pre-training from DLX-generated solutions.
Phase B: Autodidactic Iteration (ADI) — use network to guide search,
         collect new data from search outcomes, retrain.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2.nn_model import create_model, count_parameters
from phase2.data_generator import (
    enumerate_polycubes, generate_puzzle_instances,
    generate_constructive_puzzle_instances, generate_training_data,
    generate_soma_training_data, create_torch_dataset, split_dataset,
    encode_state, encode_grid, encode_placement, _check_partial_solvability,
)
from phase2.search_profiles import (
    resolve_runtime_search_settings,
    resolve_structural_fallback_settings,
)


TRAINED_MODELS_DIR = Path(__file__).parent / "trained_models"
TRAINED_MODELS_DIR.mkdir(exist_ok=True)


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, epochs=50, lr=1e-3,
          lambda_value=1.0, lambda_policy=1.0, device='cpu', verbose=True):
    """Train the CuboidNet model.

    Loss = lambda_v * BCE(value_pred, label) +
           lambda_p * CE(action_logits, target_action_index)

    Args:
        model: CuboidNet instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: number of training epochs
        lr: learning rate
        lambda_value: weight for value (solvability) loss
        lambda_policy: weight for policy (placement) loss
        device: 'cpu' or 'cuda'
        verbose: print per-epoch stats

    Returns:
        dict with training history
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    value_criterion = nn.BCELoss()
    policy_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    history = {
        'train_loss': [], 'train_value_loss': [], 'train_policy_loss': [],
        'train_value_acc': [],
        'val_loss': [], 'val_value_loss': [], 'val_policy_loss': [],
        'val_value_acc': [],
        'lr': [],
    }

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_metrics = _run_epoch(
            model, train_loader, value_criterion, policy_criterion,
            lambda_value, lambda_policy, optimizer, device
        )

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            val_metrics = _run_epoch(
                model, val_loader, value_criterion, policy_criterion,
                lambda_value, lambda_policy, None, device
            )

        scheduler.step()

        # Record history
        for key in ['loss', 'value_loss', 'policy_loss', 'value_acc']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        history['lr'].append(scheduler.get_last_lr()[0])

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0
        if verbose:
            print(f"Epoch {epoch + 1:3d}/{epochs} "
                  f"[{elapsed:.1f}s] "
                  f"train_loss={train_metrics['loss']:.4f} "
                  f"val_loss={val_metrics['loss']:.4f} "
                  f"val_acc={val_metrics['value_acc']:.3f} "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    return history


def _run_epoch(model, loader, value_criterion, policy_criterion,
               lambda_value, lambda_policy, optimizer, device):
    """Run one epoch of training or validation.

    Args:
        optimizer: if None, runs in eval mode (no gradient updates)

    Returns:
        dict with loss, value_loss, policy_loss, value_acc
    """
    is_train = optimizer is not None
    total_loss = 0.0
    total_value_loss = 0.0
    total_policy_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        states = batch['state'].to(device)
        labels = batch['label'].to(device)
        policy_candidates = batch['policy_candidates'].to(device)
        policy_action_mask = batch['policy_action_mask'].to(device)
        policy_target_idx = batch['policy_target_idx'].to(device)

        value_pred, policy_pred = model(states)
        value_pred = value_pred.squeeze(-1)

        # Value loss: BCE on solvability prediction
        v_loss = value_criterion(value_pred, labels)

        # Policy loss: CE over valid placement actions for each state.
        # Action logits are derived by projecting cell logits onto placement masks.
        action_logits = torch.bmm(
            policy_candidates, policy_pred.unsqueeze(-1)
        ).squeeze(-1)
        action_logits = action_logits.masked_fill(~policy_action_mask, -1e9)

        valid_policy = policy_target_idx >= 0
        if valid_policy.any():
            p_loss = policy_criterion(
                action_logits[valid_policy],
                policy_target_idx[valid_policy]
            )
        else:
            p_loss = torch.tensor(0.0, device=device)

        loss = lambda_value * v_loss + lambda_policy * p_loss

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * states.size(0)
        total_value_loss += v_loss.item() * states.size(0)
        total_policy_loss += p_loss.item() * states.size(0)

        # Accuracy: threshold at 0.5
        predicted = (value_pred > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += states.size(0)

    n = total if total > 0 else 1
    return {
        'loss': total_loss / n,
        'value_loss': total_value_loss / n,
        'policy_loss': total_policy_loss / n,
        'value_acc': correct / n,
    }


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(model, name, history=None, metadata=None):
    """Save model checkpoint to trained_models/ directory.

    Args:
        model: CuboidNet instance
        name: filename (without extension)
        history: training history dict
        metadata: additional info to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'in_channels': model.in_channels,
        'grid_size': model.grid_size,
        'hidden_dim': model.hidden_dim,
        'num_residual_blocks': len(model.body),
        'value_head_type': getattr(model, 'value_head_type', 'fc'),
        'policy_head_type': getattr(model, 'policy_head_type', 'fc'),
        'use_context_features': getattr(model, 'use_context_features', False),
        'gap_value_channels': getattr(model, 'gap_value_channels', 64),
        'gap_value_hidden': getattr(model, 'gap_value_hidden', 128),
        'policy_conv_inline_logit': getattr(model, 'policy_conv_inline_logit', False),
    }
    if history is not None:
        checkpoint['history'] = history
    if metadata is not None:
        checkpoint['metadata'] = metadata

    path = TRAINED_MODELS_DIR / f"{name}.pt"
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")
    return path


def _checkpoint_model_kwargs(checkpoint, state_dict):
    """Infer model-construction kwargs from a checkpoint and its weights."""
    gap_value_channels = checkpoint.get('gap_value_channels', 64)
    gap_value_hidden = checkpoint.get('gap_value_hidden', 128)
    policy_conv_inline_logit = checkpoint.get('policy_conv_inline_logit', False)

    # Backward-compatibility: early 5x5 gap/conv checkpoints used a smaller
    # gap value head and stored the policy logit conv inside policy_conv.
    if checkpoint.get('value_head_type', 'fc') == 'gap':
        gap_value_channels = state_dict['value_conv.0.weight'].shape[0]
        gap_value_hidden = state_dict['value_fc.0.weight'].shape[0]
    if (
        checkpoint.get('policy_head_type', 'fc') == 'conv'
        and 'policy_logit.weight' not in state_dict
        and 'policy_conv.3.weight' in state_dict
    ):
        policy_conv_inline_logit = True

    return {
        'grid_size': checkpoint['grid_size'],
        'max_pieces': checkpoint['in_channels'] - 1,
        'num_residual_blocks': checkpoint['num_residual_blocks'],
        'hidden_dim': checkpoint['hidden_dim'],
        'value_head_type': checkpoint.get('value_head_type', 'fc'),
        'policy_head_type': checkpoint.get('policy_head_type', 'fc'),
        'use_context_features': checkpoint.get('use_context_features', False),
        'gap_value_channels': gap_value_channels,
        'gap_value_hidden': gap_value_hidden,
        'policy_conv_inline_logit': policy_conv_inline_logit,
    }


def _transfer_state_dict(model, source_state_dict):
    """Load compatible checkpoint weights into a resized model.

    This is intended for scale-friendly gap/conv checkpoints whose stem input
    channels may grow when we increase max_pieces for larger cubes.
    """
    target_state = model.state_dict()
    merged_state = {}
    loaded_keys = []
    skipped_keys = []

    for key, target_tensor in target_state.items():
        source_tensor = source_state_dict.get(key)
        if source_tensor is None:
            skipped_keys.append(key)
            continue

        if source_tensor.shape == target_tensor.shape:
            merged_state[key] = source_tensor
            loaded_keys.append(key)
            continue

        if (
            key == 'stem.0.weight'
            and source_tensor.ndim == 5
            and target_tensor.ndim == 5
            and source_tensor.shape[0] == target_tensor.shape[0]
            and source_tensor.shape[2:] == target_tensor.shape[2:]
            and source_tensor.shape[1] <= target_tensor.shape[1]
        ):
            adapted = target_tensor.clone()
            adapted[:, :source_tensor.shape[1], :, :, :] = source_tensor

            # Initialize any extra piece channels from the mean of the learned
            # source piece-channel filters instead of random noise.
            if target_tensor.shape[1] > source_tensor.shape[1] and source_tensor.shape[1] > 1:
                extra = target_tensor.shape[1] - source_tensor.shape[1]
                piece_mean = source_tensor[:, 1:, :, :, :].mean(dim=1, keepdim=True)
                adapted[:, source_tensor.shape[1]:, :, :, :] = piece_mean.expand(
                    -1, extra, -1, -1, -1
                )
            merged_state[key] = adapted
            loaded_keys.append(key)
            continue

        skipped_keys.append(key)

    model.load_state_dict(merged_state, strict=False)
    return {
        'loaded_keys': loaded_keys,
        'skipped_keys': skipped_keys,
    }


def load_model(name, device='cpu', grid_size_override=None, max_pieces_override=None):
    """Load a model checkpoint.

    Args:
        name: filename (without extension)
        device: target device
        grid_size_override: optional target grid size for scale-transfer.
        max_pieces_override: optional target max piece-channel count.

    Returns:
        (model, history, metadata)
    """
    path = TRAINED_MODELS_DIR / f"{name}.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    model_kwargs = _checkpoint_model_kwargs(checkpoint, state_dict)
    source_grid_size = model_kwargs['grid_size']
    source_max_pieces = model_kwargs['max_pieces']
    target_grid_size = (
        source_grid_size if grid_size_override is None else int(grid_size_override)
    )
    target_max_pieces = (
        source_max_pieces if max_pieces_override is None else int(max_pieces_override)
    )

    transferring = (
        target_grid_size != source_grid_size or target_max_pieces != source_max_pieces
    )
    if transferring and (
        model_kwargs['value_head_type'] == 'fc' or model_kwargs['policy_head_type'] == 'fc'
    ):
        raise ValueError(
            f"Checkpoint '{name}' uses grid-bound fc heads and cannot be resized "
            f"from grid={source_grid_size}, max_pieces={source_max_pieces} "
            f"to grid={target_grid_size}, max_pieces={target_max_pieces}."
        )

    model = create_model(
        grid_size=target_grid_size,
        max_pieces=target_max_pieces,
        num_residual_blocks=model_kwargs['num_residual_blocks'],
        hidden_dim=model_kwargs['hidden_dim'],
        value_head_type=model_kwargs['value_head_type'],
        policy_head_type=model_kwargs['policy_head_type'],
        use_context_features=model_kwargs['use_context_features'],
        gap_value_channels=model_kwargs['gap_value_channels'],
        gap_value_hidden=model_kwargs['gap_value_hidden'],
        policy_conv_inline_logit=model_kwargs['policy_conv_inline_logit'],
    )
    transfer_info = None
    if transferring:
        transfer_info = _transfer_state_dict(model, state_dict)
    else:
        model.load_state_dict(state_dict)
    model = model.to(device)

    history = checkpoint.get('history', None)
    metadata = checkpoint.get('metadata', None)
    if transferring:
        metadata = dict(metadata or {})
        metadata['transfer_from_model'] = name
        metadata['source_grid_size'] = source_grid_size
        metadata['source_max_pieces'] = source_max_pieces
        metadata['grid_size'] = target_grid_size
        metadata['max_pieces'] = target_max_pieces
        metadata['transfer_loaded_keys'] = len(transfer_info['loaded_keys'])
        metadata['transfer_skipped_keys'] = len(transfer_info['skipped_keys'])

    return model, history, metadata


# ── Full Training Pipeline ────────────────────────────────────────────────────

def run_supervised_training(grid_size=3, max_pieces=10,
                            num_instances=100, num_negatives=2,
                            epochs=50, lr=1e-3, batch_size=64,
                            lambda_value=1.0, lambda_policy=1.0,
                            hidden_dim=128, num_residual_blocks=6,
                            value_head_type='fc', policy_head_type='fc',
                            use_context_features=False,
                            min_piece_size=3, max_piece_size=5,
                            dlx_timeout=30.0,
                            instance_source='auto',
                            constructive_variant='robust',
                            instance_seed=42,
                            device='cpu', save_name=None):
    """Run the full supervised training pipeline.

    1. Generate training data from DLX solutions
    2. Create model
    3. Train
    4. Save

    Args:
        grid_size: target cube size (3 for 3x3x3)
        max_pieces: max piece channels
        num_instances: number of puzzle instances for training
        num_negatives: negative examples per solution
        epochs: training epochs
        lr: learning rate
        batch_size: batch size
        lambda_value: weight for value loss
        lambda_policy: weight for policy loss
        hidden_dim: model hidden dimension
        num_residual_blocks: number of residual blocks
        value_head_type: 'fc' (legacy) or 'gap'
        policy_head_type: 'fc' (legacy) or 'conv'
        use_context_features: fuse global occupancy/piece stats into heads
        min_piece_size: minimum polycube size for non-3x3 data generation
        max_piece_size: maximum polycube size for non-3x3 data generation
        dlx_timeout: per-instance DLX timeout during non-3x3 data generation
        instance_source: 'auto', 'constructive', or 'dlx' for non-3x3 grids
        constructive_variant: constructive generator family
            ('mixed', 'striped', 'connected', or 'robust')
        instance_seed: seed used for non-3x3 instance generation
        device: 'cpu' or 'cuda'
        save_name: name for saved model (None to skip saving)

    Returns:
        (model, history, examples)
    """
    print("=" * 60)
    print(f"Supervised Training Pipeline (grid={grid_size}x{grid_size}x{grid_size})")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[Step 1] Generating training data...")
    if grid_size == 3:
        examples = generate_soma_training_data(
            num_instances=num_instances,
            num_negatives=num_negatives,
            max_pieces=max_pieces,
        )
    else:
        effective_source = instance_source
        if effective_source == 'auto':
            effective_source = 'constructive'

        print(f"  Generating puzzle instances (source={effective_source})...")
        if effective_source == 'constructive':
            instances = generate_constructive_puzzle_instances(
                num_instances=num_instances,
                grid_size=grid_size,
                seed=instance_seed,
                large_suite_type=constructive_variant,
                verbose=True,
            )
        elif effective_source == 'dlx':
            print("  Enumerating polycubes...")
            catalog = enumerate_polycubes(max_size=max_piece_size)
            instances = generate_puzzle_instances(
                num_instances=num_instances,
                grid_size=grid_size,
                polycube_catalog=catalog,
                min_piece_size=min_piece_size,
                max_piece_size=max_piece_size,
                dlx_timeout=dlx_timeout,
            )
        else:
            raise ValueError(
                f"Unknown instance_source='{instance_source}'. "
                "Use 'auto', 'constructive', or 'dlx'."
            )
        examples = generate_training_data(
            instances, max_pieces=max_pieces,
            num_negatives_per_solution=num_negatives,
        )

    # Step 2: Split and create datasets
    print(f"\n[Step 2] Splitting data ({len(examples)} examples)...")
    train_ex, val_ex = split_dataset(examples, group_key='instance_id')
    print(f"  Train: {len(train_ex)}, Val: {len(val_ex)}")

    train_dataset = create_torch_dataset(train_ex)
    val_dataset = create_torch_dataset(val_ex)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Step 3: Create model
    print(f"\n[Step 3] Creating model...")
    model = create_model(
        grid_size=grid_size, max_pieces=max_pieces,
        num_residual_blocks=num_residual_blocks, hidden_dim=hidden_dim,
        value_head_type=value_head_type, policy_head_type=policy_head_type,
        use_context_features=use_context_features,
    )
    total, trainable = count_parameters(model)
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    # Step 4: Train
    print(f"\n[Step 4] Training for {epochs} epochs...")
    history = train(
        model, train_loader, val_loader,
        epochs=epochs, lr=lr,
        lambda_value=lambda_value, lambda_policy=lambda_policy,
        device=device,
    )

    # Step 5: Save
    if save_name:
        print(f"\n[Step 5] Saving model...")
        metadata = {
            'grid_size': grid_size,
            'max_pieces': max_pieces,
            'num_instances': num_instances,
            'num_examples': len(examples),
            'epochs': epochs,
            'lambda_value': lambda_value,
            'lambda_policy': lambda_policy,
            'value_head_type': value_head_type,
            'policy_head_type': policy_head_type,
            'use_context_features': use_context_features,
            'instance_source': ('constructive' if grid_size > 3 and instance_source == 'auto'
                                else instance_source),
            'constructive_variant': constructive_variant,
            'instance_seed': instance_seed,
            'final_val_acc': history['val_value_acc'][-1],
        }
        save_model(model, save_name, history, metadata)

    print("\nTraining complete!")
    print(f"  Final val accuracy: {history['val_value_acc'][-1]:.3f}")
    print(f"  Best val loss: {min(history['val_loss']):.4f}")

    return model, history, examples


def run_adi_iteration(model, grid_size=3, max_pieces=10,
                      num_new_instances=50, beam_width=32,
                      adi_epochs=20, lr=5e-4, batch_size=64,
                      lambda_value=1.0, lambda_policy=1.0,
                      failed_label_mode='verify',
                      failed_verify_fraction=0.15,
                      failed_verify_max_states=16,
                      search_timeout=15.0,
                      min_piece_size=3,
                      max_piece_size=5,
                      dlx_timeout=30.0,
                      instance_source='auto',
                      constructive_variant='robust',
                      instance_seed=42,
                      adi_instances=None,
                      existing_examples=None,
                      device='cpu', verbose=True):
    """Run one round of Autodidactic Iteration (ADI).

    1. Use current model to guide beam search on new puzzle instances
    2. Record which states led to solutions vs dead ends during search
    3. Label kept-in-beam states from solved instances as positive (solvable=1.0)
    4. Label pruned/dead-end states from failed instances as negative (solvable=0.0)
    5. Combine with existing training data and retrain

    Args:
        model: pre-trained CuboidNet
        grid_size: target cube size
        max_pieces: max piece channels
        num_new_instances: number of beam search runs to perform
        beam_width: beam search width
        adi_epochs: epochs for ADI retraining
        lr: learning rate for retraining
        batch_size: batch size
        lambda_value: weight for value loss
        lambda_policy: weight for policy loss during ADI retraining
        failed_label_mode: how to handle states from failed beam searches:
            'verify' (default), 'skip', or 'negative' (legacy)
        failed_verify_fraction: fraction of failed states to verify via DLX
            when failed_label_mode='verify'
        failed_verify_max_states: max failed states to verify per instance
        search_timeout: per-instance timeout for NN beam search during ADI
        min_piece_size: min piece size for non-3x3 ADI instance generation
        max_piece_size: max piece size for non-3x3 ADI instance generation
        dlx_timeout: timeout for generating solvable ADI instances (non-3x3)
        instance_source: 'auto', 'constructive', or 'dlx' for non-3x3 ADI
        constructive_variant: constructive generator family
            ('mixed', 'striped', 'connected', or 'robust')
        instance_seed: seed used for internal non-3x3 ADI instance generation
        adi_instances: optional prebuilt list of puzzle instances; each entry
            should contain a 'pieces' field. If provided, overrides internal
            instance generation.
        existing_examples: list of training examples from supervised phase
            (combined with ADI examples for retraining). If None, train only
            on ADI-collected data.
        device: target device
        verbose: print progress

    Returns:
        (model, adi_history, new_examples)
    """
    import random as _random
    from phase2.nn_solver import nn_solve
    from phase1.test_cases import SOMA_PIECES

    print("\n" + "=" * 60)
    print("Autodidactic Iteration (ADI)")
    print("=" * 60)

    new_examples = []
    n_solved = 0
    n_failed = 0
    n_failed_labeled_neg = 0
    n_failed_verified = 0
    n_failed_skipped = 0
    zero_placement = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    # Build ADI puzzle set:
    # - 3x3 default: shuffled SOMA pieces (legacy behavior)
    # - larger grids: generate solvable instances matching target volume
    use_soma_legacy = False
    if adi_instances is not None:
        instances = list(adi_instances)
    elif grid_size == 3:
        use_soma_legacy = True
        instances = []
    else:
        effective_source = instance_source
        if effective_source == 'auto':
            effective_source = 'constructive'
        if verbose:
            print(
                f"  Preparing ADI instances for non-3x3 grid "
                f"(source={effective_source})..."
            )
        if effective_source == 'constructive':
            instances = generate_constructive_puzzle_instances(
                num_instances=num_new_instances,
                grid_size=grid_size,
                seed=instance_seed,
                large_suite_type=constructive_variant,
                verbose=verbose,
            )
        elif effective_source == 'dlx':
            catalog = enumerate_polycubes(max_size=max_piece_size)
            instances = generate_puzzle_instances(
                num_instances=num_new_instances,
                grid_size=grid_size,
                polycube_catalog=catalog,
                min_piece_size=min_piece_size,
                max_piece_size=max_piece_size,
                dlx_timeout=dlx_timeout,
                verbose=verbose,
            )
        else:
            raise ValueError(
                f"Unknown instance_source='{instance_source}'. "
                "Use 'auto', 'constructive', or 'dlx'."
            )
        if len(instances) < num_new_instances and verbose:
            print(
                f"  Warning: only {len(instances)}/{num_new_instances} ADI "
                "instances could be generated."
            )

    if use_soma_legacy:
        total_runs = num_new_instances
    else:
        total_runs = len(instances)

    if total_runs == 0:
        print("  No ADI instances available. Skipping ADI retraining.")
        return model, {}, []

    for i in range(total_runs):
        if use_soma_legacy:
            # Use SOMA_PIECES but shuffle order each time so beam search
            # explores different paths (different MRV tie-breaking).
            perm = list(range(len(SOMA_PIECES)))
            _random.shuffle(perm)
            pieces = [SOMA_PIECES[j] for j in perm]
            instance_id = f"adi_{i}"
        else:
            inst = instances[i]
            pieces = [list(p) for p in inst['pieces']]
            # Shuffle piece order to vary branching/tie-breaks while preserving puzzle.
            perm = list(range(len(pieces)))
            _random.shuffle(perm)
            pieces = [pieces[j] for j in perm]
            instance_id = f"adi_{inst.get('instance_id', i)}"

        # Run beam search WITH trace collection
        solution, trace = nn_solve(
            pieces, grid_size, model, max_pieces=max_pieces,
            beam_width=beam_width, device=device,
            return_search_trace=True, timeout=search_timeout,
        )

        if solution is not None:
            n_solved += 1
            # States kept in beam during a successful search → positive
            for entry in trace['kept']:
                new_examples.append({
                    'state': entry['state'],
                    'grid': entry['grid'],
                    'remaining_pieces': entry['remaining_pieces'],
                    'label': 1.0,  # search succeeded from here
                    'value': entry['value'],
                    'next_placement': zero_placement,
                    'next_piece_idx': -1,  # no valid policy target for ADI examples
                    'grid_size': grid_size,
                    'instance_id': instance_id,
                })
        else:
            n_failed += 1
            failed_entries = list(trace.get('pruned', [])) + list(trace.get('kept', []))

            if failed_label_mode == 'negative':
                # Legacy behavior: label all failed-search states as negative.
                for entry in failed_entries:
                    new_examples.append({
                        'state': entry['state'],
                        'grid': entry['grid'],
                        'remaining_pieces': entry['remaining_pieces'],
                        'label': 0.0,
                        'value': entry['value'],
                        'next_placement': zero_placement,
                        'next_piece_idx': -1,
                        'grid_size': grid_size,
                        'instance_id': instance_id,
                    })
                n_failed_labeled_neg += len(failed_entries)
            elif failed_label_mode == 'verify':
                added, verified, skipped = _verified_negative_examples_from_failed_trace(
                    failed_entries=failed_entries,
                    grid_size=grid_size,
                    zero_placement=zero_placement,
                    verify_fraction=failed_verify_fraction,
                    verify_max_states=failed_verify_max_states,
                    instance_id=instance_id,
                )
                new_examples.extend(added)
                n_failed_labeled_neg += len(added)
                n_failed_verified += verified
                n_failed_skipped += skipped
            elif failed_label_mode == 'skip':
                n_failed_skipped += len(failed_entries)
            else:
                raise ValueError(
                    f"Unknown failed_label_mode '{failed_label_mode}'. "
                    f"Use 'verify', 'skip', or 'negative'."
                )

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total_runs}] "
                  f"solved={n_solved} failed={n_failed} "
                  f"new_examples={len(new_examples)}")

    pos = sum(1 for e in new_examples if e['label'] == 1.0)
    neg = sum(1 for e in new_examples if e['label'] == 0.0)
    print(f"\n  ADI collection done: {len(new_examples)} examples "
          f"({pos} positive, {neg} negative)")
    print(f"  Beam search solved {n_solved}/{total_runs} instances")
    if failed_label_mode == 'verify':
        print(f"  Failed-state handling: verified={n_failed_verified}, "
              f"labeled_negative={n_failed_labeled_neg}, skipped={n_failed_skipped}")
    elif failed_label_mode == 'skip':
        print(f"  Failed-state handling: skipped={n_failed_skipped}")
    else:
        print(f"  Failed-state handling: labeled_negative={n_failed_labeled_neg}")

    if not new_examples:
        print("  No new examples generated from ADI.")
        return model, {}, new_examples

    # Combine with existing training data if provided
    all_examples = list(new_examples)
    if existing_examples:
        all_examples.extend(existing_examples)
        print(f"  Combined with {len(existing_examples)} existing examples "
              f"→ {len(all_examples)} total")

    # Retrain on combined dataset
    print(f"\n  Retraining for {adi_epochs} epochs...")
    train_ex, val_ex = split_dataset(all_examples, group_key='instance_id')
    train_dataset = create_torch_dataset(train_ex)
    val_dataset = create_torch_dataset(val_ex)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    adi_history = train(
        model, train_loader, val_loader,
        epochs=adi_epochs, lr=lr,
        lambda_value=lambda_value, lambda_policy=lambda_policy,
        device=device, verbose=verbose,
    )

    return model, adi_history, new_examples


def run_frontier_adi_iteration(
    model,
    grid_size=5,
    max_pieces=42,
    num_new_instances=40,
    adi_epochs=20,
    lr=5e-4,
    batch_size=64,
    lambda_value=1.0,
    lambda_policy=0.3,
    search_timeout=8.0,
    beam_width=None,
    max_candidates_per_state=None,
    placement_ranker=None,
    enable_pocket_pruning=None,
    max_children_per_layer=None,
    frontier_verify_roots=6,
    frontier_verify_timeout=18.0,
    frontier_verify_branch_limit=6,
    frontier_verify_max_nodes=25000,
    frontier_verify_ranker='contact',
    frontier_verify_enable_pocket_pruning=True,
    instance_source='auto',
    constructive_variant='robust',
    instance_seed=42,
    adi_instances=None,
    existing_examples=None,
    eval_callback=None,
    eval_every_epochs=0,
    early_stop_patience=0,
    device='cpu',
    verbose=True,
):
    """ADI variant that promotes only frontier states verified solvable by deeper search."""
    import random as _random
    from phase2.nn_solver import nn_solve, dfs_solve_from_frontier
    from phase1.test_cases import SOMA_PIECES

    print("\n" + "=" * 60)
    print("Frontier-Verified ADI")
    print("=" * 60)

    runtime = resolve_runtime_search_settings(
        grid_size,
        beam_width=beam_width,
        max_candidates_per_state=max_candidates_per_state,
        placement_ranker=placement_ranker,
        enable_pocket_pruning=enable_pocket_pruning,
        max_children_per_layer=max_children_per_layer,
        beam_diversity_slots=None,
        beam_diversity_metric=None,
        default_beam_width=32,
        default_max_candidates_per_state=50,
    )
    structural = resolve_structural_fallback_settings(grid_size) or {}
    if frontier_verify_branch_limit is None:
        frontier_verify_branch_limit = structural.get("dfs_branch_limit", 6)
    if frontier_verify_max_nodes is None:
        frontier_verify_max_nodes = structural.get("dfs_max_nodes", 25000)
    if frontier_verify_timeout is None:
        frontier_verify_timeout = structural.get("dfs_timeout", 18.0)

    zero_placement = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    new_examples = []
    n_solved = 0
    n_failed = 0
    n_frontier_verified = 0

    use_soma_legacy = False
    if adi_instances is not None:
        instances = list(adi_instances)
    elif grid_size == 3:
        use_soma_legacy = True
        instances = []
    else:
        effective_source = instance_source
        if effective_source == 'auto':
            effective_source = 'constructive'
        if verbose:
            print(
                f"  Preparing frontier-ADI instances (source={effective_source})..."
            )
        if effective_source == 'constructive':
            instances = generate_constructive_puzzle_instances(
                num_instances=num_new_instances,
                grid_size=grid_size,
                seed=instance_seed,
                large_suite_type=constructive_variant,
                verbose=verbose,
            )
        elif effective_source == 'dlx':
            catalog = enumerate_polycubes(max_size=5)
            instances = generate_puzzle_instances(
                num_instances=num_new_instances,
                grid_size=grid_size,
                polycube_catalog=catalog,
                min_piece_size=3,
                max_piece_size=5,
                dlx_timeout=30.0,
                verbose=verbose,
            )
        else:
            raise ValueError(
                f"Unknown instance_source='{instance_source}'. "
                "Use 'auto', 'constructive', or 'dlx'."
            )

    total_runs = num_new_instances if use_soma_legacy else len(instances)
    if total_runs == 0:
        print("  No frontier-ADI instances available. Skipping.")
        return model, {}, []

    for i in range(total_runs):
        if use_soma_legacy:
            perm = list(range(len(SOMA_PIECES)))
            _random.shuffle(perm)
            pieces = [SOMA_PIECES[j] for j in perm]
            instance_id = f"frontier_adi_{i}"
        else:
            inst = instances[i]
            pieces = [list(p) for p in inst['pieces']]
            perm = list(range(len(pieces)))
            _random.shuffle(perm)
            pieces = [pieces[j] for j in perm]
            instance_id = f"frontier_adi_{inst.get('instance_id', i)}"

        beam_frontier = []
        solution, trace = nn_solve(
            pieces,
            grid_size,
            model,
            max_pieces=max_pieces,
            beam_width=runtime["beam_width"],
            timeout=search_timeout,
            device=device,
            return_search_trace=True,
            max_candidates_per_state=runtime["max_candidates_per_state"],
            enable_pocket_pruning=runtime["enable_pocket_pruning"],
            placement_ranker=runtime["placement_ranker"],
            max_children_per_layer=runtime["max_children_per_layer"],
            beam_diversity_slots=runtime["beam_diversity_slots"],
            beam_diversity_metric=runtime["beam_diversity_metric"],
            frontier_out=beam_frontier,
        )

        if solution is not None:
            n_solved += 1
            for entry in trace['kept']:
                new_examples.append({
                    'state': entry['state'],
                    'grid': entry['grid'],
                    'remaining_pieces': entry['remaining_pieces'],
                    'label': 1.0,
                    'value': entry['value'],
                    'next_placement': zero_placement,
                    'next_piece_idx': -1,
                    'policy_candidates': np.zeros((0, grid_size ** 3), dtype=np.float32),
                    'policy_target_idx': -1,
                    'grid_size': grid_size,
                    'instance_id': instance_id,
                })
        else:
            n_failed += 1
            roots = beam_frontier[:max(1, frontier_verify_roots)]
            per_root_timeout = max(1.0, frontier_verify_timeout / max(1, len(roots)))
            per_root_nodes = max(1000, int(frontier_verify_max_nodes / max(1, len(roots))))

            for root_idx, root in enumerate(roots):
                root_solution, _ = dfs_solve_from_frontier(
                    pieces=pieces,
                    grid_size=grid_size,
                    model=model,
                    frontier_states=[root],
                    max_pieces=max_pieces,
                    timeout=per_root_timeout,
                    max_frontier_states=1,
                    branch_limit=frontier_verify_branch_limit,
                    max_nodes=per_root_nodes,
                    device=device,
                    placement_ranker=frontier_verify_ranker,
                    enable_pocket_pruning=frontier_verify_enable_pocket_pruning,
                    use_transposition=True,
                )
                if root_solution is None:
                    continue

                root_state = encode_state(
                    root['grid'],
                    root['remaining_pieces'],
                    grid_size,
                    max_pieces,
                )
                new_examples.append({
                    'state': root_state,
                    'grid': root['grid'],
                    'remaining_pieces': root['remaining_pieces'],
                    'label': 1.0,
                    'value': len(root['remaining_pieces']),
                    'next_placement': zero_placement,
                    'next_piece_idx': -1,
                    'policy_candidates': np.zeros((0, grid_size ** 3), dtype=np.float32),
                    'policy_target_idx': -1,
                    'grid_size': grid_size,
                    'instance_id': f"{instance_id}_root{root_idx}",
                })
                n_frontier_verified += 1

        if verbose and (i + 1) % 10 == 0:
            print(
                f"  [{i+1}/{total_runs}] solved={n_solved} failed={n_failed} "
                f"frontier_verified={n_frontier_verified} new_examples={len(new_examples)}"
            )

    pos = sum(1 for e in new_examples if e['label'] == 1.0)
    neg = sum(1 for e in new_examples if e['label'] == 0.0)
    print(f"\n  Frontier ADI collection done: {len(new_examples)} examples "
          f"({pos} positive, {neg} negative)")
    print(f"  Beam search solved {n_solved}/{total_runs} instances")
    print(f"  Frontier roots verified solvable: {n_frontier_verified}")

    if not new_examples:
        print("  No new frontier-ADI examples generated.")
        return model, {}, new_examples

    all_examples = list(new_examples)
    if existing_examples:
        all_examples.extend(existing_examples)
        print(f"  Combined with {len(existing_examples)} existing examples "
              f"-> {len(all_examples)} total")

    print(f"\n  Retraining for {adi_epochs} epochs...")
    train_ex, val_ex = split_dataset(all_examples, group_key='instance_id')
    train_dataset = create_torch_dataset(train_ex)
    val_dataset = create_torch_dataset(val_ex)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    adi_history = {}
    proxy_eval_history = []
    best_proxy_score = None
    best_proxy_state = None
    no_improve = 0
    epochs_completed = 0
    chunk_size = int(eval_every_epochs) if eval_callback and eval_every_epochs else int(adi_epochs)
    chunk_size = max(1, chunk_size)

    def _extend_history(dst, src):
        for key, values in src.items():
            dst.setdefault(key, [])
            dst[key].extend(values)

    while epochs_completed < int(adi_epochs):
        this_chunk = min(chunk_size, int(adi_epochs) - epochs_completed)
        chunk_history = train(
            model, train_loader, val_loader,
            epochs=this_chunk, lr=lr,
            lambda_value=lambda_value, lambda_policy=lambda_policy,
            device=device, verbose=verbose,
        )
        _extend_history(adi_history, chunk_history)
        epochs_completed += this_chunk

        if eval_callback is None:
            continue

        proxy_summary = eval_callback(model=model, epochs_trained=epochs_completed)
        if proxy_summary is None:
            continue
        proxy_summary = dict(proxy_summary)
        proxy_summary['epochs_trained'] = epochs_completed
        proxy_eval_history.append(proxy_summary)

        proxy_score = proxy_summary.get('score')
        if proxy_score is None:
            continue

        improved = best_proxy_score is None or proxy_score > best_proxy_score
        if improved:
            best_proxy_score = proxy_score
            best_proxy_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        if verbose:
            worst_bucket = proxy_summary.get('worst_bucket_solve_rate')
            print(
                f"  Proxy eval @ epoch {epochs_completed}: "
                f"score={proxy_score:.4f} "
                f"worst_bucket={worst_bucket if worst_bucket is not None else 'n/a'}"
            )

        if early_stop_patience and no_improve >= int(early_stop_patience):
            if verbose:
                print(
                    f"  Early stopping after {epochs_completed} epochs "
                    f"(proxy plateau: patience={early_stop_patience})"
                )
            break

    if best_proxy_state is not None:
        model.load_state_dict(best_proxy_state)
        model = model.to(device)

    if proxy_eval_history:
        adi_history['proxy_eval'] = proxy_eval_history
    adi_history['_epochs_completed'] = epochs_completed
    if best_proxy_score is not None:
        adi_history['_best_proxy_score'] = best_proxy_score

    return model, adi_history, new_examples


# ── ADI Verification Helpers ─────────────────────────────────────────────────

def _verified_negative_examples_from_failed_trace(
    failed_entries, grid_size, zero_placement, verify_fraction, verify_max_states,
    instance_id
):
    """Create negatives only from failed states verified unsolvable by DLX."""
    if not failed_entries:
        return [], 0, 0

    verify_fraction = max(0.0, min(1.0, verify_fraction))
    sample_n = int(len(failed_entries) * verify_fraction)
    if verify_fraction > 0.0 and sample_n == 0:
        sample_n = 1
    sample_n = min(sample_n, verify_max_states, len(failed_entries))
    if sample_n <= 0:
        return [], 0, len(failed_entries)

    sampled = list(failed_entries)
    np.random.shuffle(sampled)
    sampled = sampled[:sample_n]

    added = []
    for entry in sampled:
        if _is_state_unsolvable(entry, grid_size):
            added.append({
                'state': entry['state'],
                'grid': entry['grid'],
                'remaining_pieces': entry['remaining_pieces'],
                'label': 0.0,
                'value': entry['value'],
                'next_placement': zero_placement,
                'next_piece_idx': -1,
                'grid_size': grid_size,
                'instance_id': instance_id,
            })

    verified = sample_n
    skipped = len(failed_entries) - sample_n
    return added, verified, skipped


def _is_state_unsolvable(entry, grid_size):
    """Return True iff remaining pieces cannot fill current empty cells."""
    grid = entry['grid']
    remaining_pieces = entry['remaining_pieces']

    occupied = set(tuple(idx) for idx in np.argwhere(grid > 0.5))
    empty_cells = set(
        (x, y, z)
        for x in range(grid_size)
        for y in range(grid_size)
        for z in range(grid_size)
        if (x, y, z) not in occupied
    )

    # Mismatched volume implies unsolvable under current encoding.
    remaining_volume = sum(len(p) for p in remaining_pieces)
    if remaining_volume != len(empty_cells):
        return True

    solvable = _check_partial_solvability(remaining_pieces, empty_cells, grid_size)
    return not solvable


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_training_curves(history, title="Training Curves"):
    """Plot training and validation loss/accuracy curves.

    Args:
        history: dict from train()
        title: figure title

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Value accuracy
    axes[1].plot(epochs, history['train_value_acc'], label='Train')
    axes[1].plot(epochs, history['val_value_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Value (Solvability) Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.05)

    # Learning rate
    axes[2].plot(epochs, history['lr'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Quick training run on Soma cube data
    model, history, examples = run_supervised_training(
        grid_size=3, max_pieces=10,
        num_instances=50, num_negatives=2,
        epochs=30, lr=1e-3, batch_size=32,
        hidden_dim=64, num_residual_blocks=4,
        save_name="soma_3x3x3_quick",
    )
