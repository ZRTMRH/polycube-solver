"""
Phase 1: DLX Exact Cover Solver for polycube packing problems.
"""

from .polycube import (
    normalize, rotate, get_orientations, get_all_placements, ROTATIONS,
)
from .dlx_solver import DLX
from .solver import solve, cube_root_int
from .visualization import plot_solution, animate_solution, plot_pieces, build_voxel_grid
from .test_cases import SOMA_PIECES, verify_solution
