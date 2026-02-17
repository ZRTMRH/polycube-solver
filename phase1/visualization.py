"""
3D visualization of polycube solutions using matplotlib voxel plots.
Supports piece-by-piece animation and GIF export.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.colors as mcolors


# Distinct colors for up to ~20 pieces
PIECE_COLORS = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
]


def build_voxel_grid(solution, grid_size):
    """Build a voxel grid and color array from a solution.

    Args:
        solution: dict mapping piece_index -> frozenset of (x, y, z) cells
        grid_size: int, side length of the cube

    Returns:
        (filled, colors): both are NxNxN numpy arrays.
            filled[x,y,z] = True if cell is occupied
            colors[x,y,z] = color string for that cell's piece
    """
    filled = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    colors = np.empty((grid_size, grid_size, grid_size), dtype=object)

    for piece_idx, cells in solution.items():
        color = PIECE_COLORS[piece_idx % len(PIECE_COLORS)]
        for x, y, z in cells:
            filled[x, y, z] = True
            colors[x, y, z] = color

    return filled, colors


def plot_solution(solution, grid_size, title="Polycube Solution", ax=None):
    """Plot a complete solution as colored voxels.

    Args:
        solution: dict mapping piece_index -> frozenset of (x, y, z) cells
        grid_size: int
        title: plot title
        ax: optional matplotlib 3D axes; created if None

    Returns:
        (fig, ax) tuple
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    filled, colors = build_voxel_grid(solution, grid_size)

    # Convert colors to RGBA facecolors array
    facecolors = np.empty(filled.shape + (4,), dtype=float)
    edgecolors = np.empty(filled.shape + (4,), dtype=float)
    for idx in np.ndindex(filled.shape):
        if filled[idx]:
            rgba = mcolors.to_rgba(colors[idx], alpha=0.85)
            facecolors[idx] = rgba
            edgecolors[idx] = (0, 0, 0, 0.3)
        else:
            facecolors[idx] = (0, 0, 0, 0)
            edgecolors[idx] = (0, 0, 0, 0)

    ax.voxels(filled, facecolors=facecolors, edgecolors=edgecolors)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Equal aspect ratio
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_zlim(0, grid_size)

    return fig, ax


def animate_solution(solution, grid_size, title="Polycube Assembly",
                     interval=800, save_path=None):
    """Create a piece-by-piece animation of the solution assembly.

    Args:
        solution: dict mapping piece_index -> frozenset of (x, y, z) cells
        grid_size: int
        title: animation title
        interval: milliseconds between frames
        save_path: if provided, save as GIF to this path

    Returns:
        FuncAnimation object
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Sort pieces by index for consistent ordering
    piece_order = sorted(solution.keys())
    n_pieces = len(piece_order)

    def update(frame):
        ax.cla()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_zlim(0, grid_size)
        ax.set_title(f"{title} — piece {frame + 1}/{n_pieces}")

        # Build partial solution up to current frame
        partial = {k: solution[k] for k in piece_order[:frame + 1]}
        filled, colors = build_voxel_grid(partial, grid_size)

        facecolors = np.empty(filled.shape + (4,), dtype=float)
        edgecolors = np.empty(filled.shape + (4,), dtype=float)
        for idx in np.ndindex(filled.shape):
            if filled[idx]:
                rgba = mcolors.to_rgba(colors[idx], alpha=0.85)
                facecolors[idx] = rgba
                edgecolors[idx] = (0, 0, 0, 0.3)
            else:
                facecolors[idx] = (0, 0, 0, 0)
                edgecolors[idx] = (0, 0, 0, 0)

        ax.voxels(filled, facecolors=facecolors, edgecolors=edgecolors)
        return []

    anim = FuncAnimation(fig, update, frames=n_pieces,
                         interval=interval, repeat=True)

    if save_path:
        writer = PillowWriter(fps=max(1, 1000 // interval))
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")

    return anim


def plot_pieces(pieces, title="Polycube Pieces"):
    """Plot individual pieces in a grid layout.

    Args:
        pieces: list of pieces, each a list of (x, y, z) tuples
        title: figure title
    """
    n = len(pieces)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    fig.suptitle(title, fontsize=14)

    for i, piece in enumerate(pieces):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

        # Build a small voxel grid for this piece
        coords = list(piece)
        max_c = max(max(c) for c in zip(*coords)) + 1 if coords else 1
        size = max_c

        filled = np.zeros((size, size, size), dtype=bool)
        color = PIECE_COLORS[i % len(PIECE_COLORS)]

        for x, y, z in coords:
            filled[x, y, z] = True

        facecolors = np.where(filled[..., np.newaxis],
                              np.array(mcolors.to_rgba(color, alpha=0.85)),
                              np.array([0, 0, 0, 0]))
        edgecolors = np.where(filled[..., np.newaxis],
                              np.array([0, 0, 0, 0.3]),
                              np.array([0, 0, 0, 0]))

        ax.voxels(filled, facecolors=facecolors, edgecolors=edgecolors)
        ax.set_title(f"Piece {i} ({len(piece)} cubes)")
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_zlim(0, size)

    plt.tight_layout()
    return fig
