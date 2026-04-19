"""
CuboidNet: 3D ResNet for polycube packing state evaluation.

Architecture inspired by DeepCube (Agostinelli et al., 2019) and AlphaGo,
adapted for 3D voxel grids. The network has:
- A shared 3D CNN trunk with residual blocks
- A value head: predicts P(solvable | state) via sigmoid
- A policy head: predicts placement logits for each remaining piece
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 hidden_dim=128):
        super().__init__()
        self.grid_size = grid_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

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

        # Value head: predict P(solvable)
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

        # Policy head: predict spatial placement logits
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

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, C, N, N, N) input tensor

        Returns:
            value: (batch, 1) solvability probability (sigmoid applied)
            policy: (batch, N^3) placement logits (raw, no softmax)
        """
        # Shared trunk
        h = self.stem(x)
        h = self.body(h)

        # Value head
        v = self.value_conv(h)
        v = v.view(v.size(0), -1)
        value = torch.sigmoid(self.value_fc(v))

        # Policy head
        p = self.policy_conv(h)
        p = p.view(p.size(0), -1)
        policy = self.policy_fc(p)

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
                 hidden_dim=128):
    """Factory function to create a CuboidNet model.

    Args:
        grid_size: side length of the target cube
        max_pieces: maximum number of pieces (determines input channels)
        num_residual_blocks: depth of residual trunk
        hidden_dim: width of hidden layers

    Returns:
        CuboidNet model
    """
    in_channels = 1 + max_pieces  # grid occupancy + piece channels
    model = CuboidNet(
        in_channels=in_channels,
        grid_size=grid_size,
        num_residual_blocks=num_residual_blocks,
        hidden_dim=hidden_dim,
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
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")

    
    # Test forward pass

    """x = torch.randn(1, model.in_channels, grid_size, grid_size, grid_size)
    print ("not the torch")
    value, policy = model(x)"""
    was_training = model.training
    model.eval()
    with torch.no_grad():
            x = torch.randn(1, model.in_channels, grid_size, grid_size, grid_size)
            print("model(x) is the prob")
            value, policy = model(x)
    if was_training:

        model.train()  # restore original mode
   
    print(f"  Value output shape: {value.shape}")
    print(f"  Policy output shape: {policy.shape}")
    return total, trainable


if __name__ == "__main__":
    # Test the model
    print("Testing CuboidNet for 3x3x3 grid:")
    model = create_model(grid_size=3, max_pieces=10, num_residual_blocks=6, hidden_dim=128)
    model_summary(model, grid_size=3, max_pieces=10)

    print("\nTesting CuboidNet for 4x4x4 grid:")
    model4 = create_model(grid_size=4, max_pieces=15, num_residual_blocks=6, hidden_dim=128)
    model_summary(model4, grid_size=4, max_pieces=15)
