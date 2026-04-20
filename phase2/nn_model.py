"""
CuboidNet: 3D ResNet for polycube packing state evaluation.

Architecture inspired by DeepCube (Agostinelli et al., 2019) and AlphaGo,
adapted for 3D voxel grids. The network has:
- A shared 3D CNN trunk with residual blocks
- A value head: predicts P(solvable | state) via sigmoid
- A policy head: predicts placement logits for each remaining piece

Head variants
-------------
- value_head_type='fc'  (legacy): flat FC over the full spatial map.
                                   Weight shape depends on grid_size.
- value_head_type='gap' (scale-invariant): global-average-pool the conv
                                   trunk, then a small FC. Works at any N.

- policy_head_type='fc'   (legacy): flat FC mapping spatial features to
                                    grid_size**3 logits. Grid-size bound.
- policy_head_type='conv' (scale-invariant): 1x1x1 conv producing one
                                    logit per voxel, then flattened.

Context features
----------------
When use_context_features=True, a handful of global scalars are extracted
from the input tensor (fill ratio, remaining volume ratio, piece-count
ratio, average remaining-piece size) and projected to a small feature
vector. The projection is concatenated into the value head (gap variant)
and broadcast-concatenated into the policy head (conv variant).
Context features are ignored by the legacy 'fc' heads.

Backward compatibility
----------------------
With defaults (value_head_type='fc', policy_head_type='fc',
use_context_features=False), the module structure is byte-identical to
the pre-refactor architecture, so checkpoints saved before the head-type
refactor load unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Small fixed-size context feature vector extracted from the state tensor.
# Keep this constant so a saved checkpoint's context_proj weights have a
# stable input dimension.
CONTEXT_INPUT_DIM = 4
CONTEXT_PROJ_DIM = 32


class ResBlock3d(nn.Module):
    """3D Residual block: Conv3d -> BN -> ReLU -> Conv3d -> BN -> skip -> ReLU."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


def _extract_context_features(x):
    """Extract a small scale-invariant summary of the state tensor.

    Args:
        x: (B, 1 + max_pieces, N, N, N) input state.
           Channel 0 is the occupancy grid, channels 1: are per-piece
           cell-indicator volumes.

    Returns:
        (B, CONTEXT_INPUT_DIM) float tensor of scale-invariant summaries.
    """
    B = x.size(0)
    N = x.size(-1)
    voxels = float(N * N * N)
    max_pieces = x.size(1) - 1

    occupancy = x[:, 0]                    # (B, N, N, N)
    piece_chans = x[:, 1:]                 # (B, P, N, N, N)

    fill_ratio = occupancy.reshape(B, -1).mean(dim=1)

    piece_volumes = piece_chans.reshape(B, max_pieces, -1).sum(dim=2)  # (B, P)
    total_remaining = piece_volumes.sum(dim=1)                         # (B,)
    remaining_volume_ratio = total_remaining / voxels

    num_remaining = (piece_volumes > 0).sum(dim=1).float()             # (B,)
    piece_count_ratio = num_remaining / max(max_pieces, 1)

    avg_piece_size = total_remaining / num_remaining.clamp(min=1.0)
    # Normalize avg size by a reasonable piece-size scale (max polycube size 5)
    # so it sits in roughly [0, 1]. Exact constant doesn't matter; consistency does.
    avg_piece_size = avg_piece_size / 5.0

    return torch.stack(
        [fill_ratio, remaining_volume_ratio, piece_count_ratio, avg_piece_size],
        dim=1,
    )


class CuboidNet(nn.Module):
    """3D CNN with residual blocks for polycube state evaluation.

    Input: (batch, C, N, N, N) where
        C = 1 (grid occupancy) + max_pieces (remaining piece channels)
        N = grid_size

    Output:
        value: (batch, 1) — P(solvable | state), via sigmoid
        policy: (batch, N*N*N) — placement logits (flat spatial distribution)
    """

    def __init__(self, in_channels, grid_size, num_residual_blocks=6,
                 hidden_dim=128,
                 value_head_type='fc', policy_head_type='fc',
                 use_context_features=False,
                 gap_value_channels=64,
                 gap_value_hidden=128,
                 policy_conv_inline_logit=False):
        super().__init__()
        if value_head_type not in ('fc', 'gap'):
            raise ValueError(f"value_head_type must be 'fc' or 'gap', got {value_head_type!r}")
        if policy_head_type not in ('fc', 'conv'):
            raise ValueError(f"policy_head_type must be 'fc' or 'conv', got {policy_head_type!r}")
        if policy_conv_inline_logit and policy_head_type != 'conv':
            raise ValueError("policy_conv_inline_logit is only valid for policy_head_type='conv'")
        if policy_conv_inline_logit and use_context_features:
            raise ValueError(
                "policy_conv_inline_logit is incompatible with use_context_features=True"
            )

        self.grid_size = grid_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.value_head_type = value_head_type
        self.policy_head_type = policy_head_type
        self.use_context_features = use_context_features
        self.gap_value_channels = gap_value_channels
        self.gap_value_hidden = gap_value_hidden
        self.policy_conv_inline_logit = policy_conv_inline_logit

        # Stem: project input channels to hidden_dim
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
        )

        # Body: residual blocks
        self.body = nn.Sequential(
            *[ResBlock3d(hidden_dim) for _ in range(num_residual_blocks)]
        )

        # Optional context feature projection
        if use_context_features:
            self.context_proj = nn.Sequential(
                nn.Linear(CONTEXT_INPUT_DIM, CONTEXT_PROJ_DIM),
                nn.ReLU(),
                nn.Linear(CONTEXT_PROJ_DIM, CONTEXT_PROJ_DIM),
                nn.ReLU(),
            )
            ctx_dim = CONTEXT_PROJ_DIM
        else:
            self.context_proj = None
            ctx_dim = 0

        # ── Value head ────────────────────────────────────────────────
        if value_head_type == 'fc':
            # Legacy: flat FC over the full spatial map. Grid-size bound.
            self.value_conv = nn.Sequential(
                nn.Conv3d(hidden_dim, 32, kernel_size=1, bias=False),
                nn.BatchNorm3d(32),
                nn.ReLU(),
            )
            self.value_fc = nn.Sequential(
                nn.Linear(32 * grid_size ** 3, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        else:  # 'gap'
            # Scale-invariant: 1x1 reduce, global-average-pool, then small FC.
            self.value_conv = nn.Sequential(
                nn.Conv3d(hidden_dim, gap_value_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(gap_value_channels),
                nn.ReLU(),
            )
            self.value_fc = nn.Sequential(
                nn.Linear(gap_value_channels + ctx_dim, gap_value_hidden),
                nn.ReLU(),
                nn.Linear(gap_value_hidden, 1),
            )

        # ── Policy head ───────────────────────────────────────────────
        if policy_head_type == 'fc':
            # Legacy: flat FC to grid_size**3 logits. Grid-size bound.
            self.policy_conv = nn.Sequential(
                nn.Conv3d(hidden_dim, 64, kernel_size=1, bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(),
            )
            self.policy_fc = nn.Sequential(
                nn.Linear(64 * grid_size ** 3, 512),
                nn.ReLU(),
                nn.Linear(512, grid_size ** 3),
            )
        else:  # 'conv'
            # Scale-invariant: 1x1 reduce, optional broadcast-concat of
            # context, then 1x1 conv to per-voxel logits.
            if policy_conv_inline_logit:
                # Legacy scale-invariant conv head used by early 5x5 checkpoints.
                self.policy_conv = nn.Sequential(
                    nn.Conv3d(hidden_dim, 64, kernel_size=1, bias=False),
                    nn.BatchNorm3d(64),
                    nn.ReLU(),
                    nn.Conv3d(64, 1, kernel_size=1),
                )
                self.policy_logit = None
            else:
                self.policy_conv = nn.Sequential(
                    nn.Conv3d(hidden_dim, 64, kernel_size=1, bias=False),
                    nn.BatchNorm3d(64),
                    nn.ReLU(),
                )
                self.policy_logit = nn.Conv3d(64 + ctx_dim, 1, kernel_size=1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, C, N, N, N) input tensor

        Returns:
            value:  (batch, 1)     sigmoid-squashed solvability probability
            policy: (batch, N^3)   placement logits (raw, no softmax)
        """
        # Shared trunk
        h = self.stem(x)
        h = self.body(h)

        # Context features (if enabled)
        if self.use_context_features:
            ctx_feats = _extract_context_features(x)          # (B, CONTEXT_INPUT_DIM)
            ctx_embed = self.context_proj(ctx_feats)          # (B, CONTEXT_PROJ_DIM)
        else:
            ctx_embed = None

        # ── Value head ────────────────────────────────────────────────
        v = self.value_conv(h)
        if self.value_head_type == 'fc':
            v = v.view(v.size(0), -1)
            value = torch.sigmoid(self.value_fc(v))
        else:  # 'gap'
            # Global average pool over spatial dims: (B, C, N, N, N) -> (B, C)
            v_pooled = v.mean(dim=(2, 3, 4))
            if ctx_embed is not None:
                v_pooled = torch.cat([v_pooled, ctx_embed], dim=1)
            value = torch.sigmoid(self.value_fc(v_pooled))

        # ── Policy head ───────────────────────────────────────────────
        p = self.policy_conv(h)
        if self.policy_head_type == 'fc':
            p = p.view(p.size(0), -1)
            policy = self.policy_fc(p)
        else:  # 'conv'
            if self.policy_conv_inline_logit:
                policy = p.view(p.size(0), -1)
            else:
                if ctx_embed is not None:
                # Broadcast the context vector across every voxel and concat.
                    B, _, N1, N2, N3 = p.shape
                    ctx_map = ctx_embed.view(B, -1, 1, 1, 1).expand(B, -1, N1, N2, N3)
                    p = torch.cat([p, ctx_map], dim=1)
                logit_map = self.policy_logit(p)                  # (B, 1, N, N, N)
                policy = logit_map.view(logit_map.size(0), -1)    # (B, N^3)

        return value, policy

    def predict_value(self, x):
        """Predict only the value (solvability) — useful during search."""
        with torch.no_grad():
            value, _ = self.forward(x)
        return value

    def predict_policy(self, x):
        """Predict only the policy (placement logits) — useful during search."""
        with torch.no_grad():
            _, policy = self.forward(x)
        return policy


def create_model(grid_size=3, max_pieces=10, num_residual_blocks=6,
                 hidden_dim=128,
                 value_head_type='fc', policy_head_type='fc',
                 use_context_features=False,
                 gap_value_channels=64,
                 gap_value_hidden=128,
                 policy_conv_inline_logit=False):
    """Factory function to create a CuboidNet model.

    Args:
        grid_size: side length of the target cube
        max_pieces: maximum number of pieces (determines input channels)
        num_residual_blocks: depth of residual trunk
        hidden_dim: width of hidden layers
        value_head_type: 'fc' (legacy, grid-size bound) or 'gap' (scale-invariant)
        policy_head_type: 'fc' (legacy, grid-size bound) or 'conv' (scale-invariant)
        use_context_features: fuse global occupancy/piece stats into the heads.
            Only affects 'gap'/'conv' heads (ignored by 'fc' heads, but the
            context_proj module is still created so checkpoints remain consistent).
        gap_value_channels: output width of the gap value reduction conv.
        gap_value_hidden: hidden width of the gap value MLP.
        policy_conv_inline_logit: use the legacy inline 1x1 policy conv head.

    Returns:
        CuboidNet model
    """
    in_channels = 1 + max_pieces  # grid occupancy + piece channels
    model = CuboidNet(
        in_channels=in_channels,
        grid_size=grid_size,
        num_residual_blocks=num_residual_blocks,
        hidden_dim=hidden_dim,
        value_head_type=value_head_type,
        policy_head_type=policy_head_type,
        use_context_features=use_context_features,
        gap_value_channels=gap_value_channels,
        gap_value_hidden=gap_value_hidden,
        policy_conv_inline_logit=policy_conv_inline_logit,
    )
    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary(model, grid_size=3, max_pieces=10):
    """Print a summary of the model architecture."""
    total, trainable = count_parameters(model)
    print(f"CuboidNet Summary:")
    print(f"  Grid size: {grid_size}x{grid_size}x{grid_size}")
    print(f"  Input channels: {model.in_channels} (1 grid + {max_pieces} pieces)")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Residual blocks: {len(model.body)}")
    print(f"  Value head: {model.value_head_type}")
    print(f"  Policy head: {model.policy_head_type}")
    print(f"  Context features: {model.use_context_features}")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    # Test forward pass
    x = torch.randn(1, model.in_channels, grid_size, grid_size, grid_size)
    value, policy = model(x)
    print(f"  Value output shape: {value.shape}")
    print(f"  Policy output shape: {policy.shape}")
    return total, trainable


if __name__ == "__main__":
    # Test the model
    print("Testing CuboidNet for 3x3x3 grid (legacy fc/fc):")
    model = create_model(grid_size=3, max_pieces=10, num_residual_blocks=6, hidden_dim=128)
    model_summary(model, grid_size=3, max_pieces=10)

    print("\nTesting CuboidNet for 4x4x4 grid (gap/conv + context):")
    model4 = create_model(
        grid_size=4, max_pieces=15, num_residual_blocks=6, hidden_dim=128,
        value_head_type='gap', policy_head_type='conv', use_context_features=True,
    )
    model_summary(model4, grid_size=4, max_pieces=15)

    print("\nTesting scale-invariance: load gap/conv at N=4, apply at N=7")
    x7 = torch.randn(2, model4.in_channels, 7, 7, 7)
    v7, p7 = model4(x7)
    print(f"  N=7 value shape: {v7.shape}, policy shape: {p7.shape} (should be (2, 343))")
