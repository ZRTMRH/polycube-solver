"""
Main solver: formulate the polycube packing problem as an exact cover
instance and solve it with DLX.
"""

from .polycube import get_all_placements
from .dlx_solver import DLX


def cube_root_int(n):
    """Return integer cube root of n if n is a perfect cube, else None."""
    if n <= 0:
        return None
    cr = round(n ** (1.0 / 3.0))
    # Check neighbors to handle floating-point rounding
    for candidate in (cr - 1, cr, cr + 1):
        if candidate >= 0 and candidate ** 3 == n:
            return candidate
    return None


def solve(pieces, grid_size=None, find_all=False):
    """Solve the polycube packing problem.

    Given a list of polycube pieces, determine if they can be packed into
    a perfect cube with no gaps and no overlaps.

    Args:
        pieces: list of pieces, each piece is a list/set of (x, y, z) tuples
        grid_size: side length of target cube (auto-detected from volume if None)
        find_all: if True, return all solutions

    Returns:
        list of solutions, each solution is a dict mapping piece_index -> frozenset of cells,
        or empty list if no solution exists
    """
    # Compute total volume
    total_volume = sum(len(p) for p in pieces)

    # Determine grid size
    if grid_size is None:
        grid_size = cube_root_int(total_volume)
        if grid_size is None:
            print(f"Total volume {total_volume} is not a perfect cube.")
            return []
    else:
        if total_volume != grid_size ** 3:
            print(f"Total volume {total_volume} != {grid_size}^3 = {grid_size ** 3}")
            return []

    print(f"Target: {grid_size}x{grid_size}x{grid_size} cube "
          f"({total_volume} cells, {len(pieces)} pieces)")

    # Generate all placements for each piece
    all_placements = []  # list of (piece_idx, placement_frozenset)
    for i, piece in enumerate(pieces):
        placements = get_all_placements(piece, grid_size)
        if not placements:
            print(f"Piece {i} ({piece}) has no valid placements!")
            return []
        all_placements.append((i, placements))
        print(f"  Piece {i}: {len(piece)} cubes, {len(placements)} placements")

    # Build exact cover columns
    # Cell columns: one per cell in the grid
    cell_names = []
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                cell_names.append(f"c_{x}_{y}_{z}")

    # Piece columns: one per piece (each piece used exactly once)
    piece_names = [f"p_{i}" for i in range(len(pieces))]

    all_columns = cell_names + piece_names
    dlx = DLX(all_columns)

    # Add rows: one per valid placement of each piece
    row_id = 0
    row_map = {}  # row_id -> (piece_idx, placement)
    for piece_idx, placements in all_placements:
        for placement in placements:
            cols = [f"p_{piece_idx}"]
            for x, y, z in placement:
                cols.append(f"c_{x}_{y}_{z}")
            dlx.add_row(row_id, cols)
            row_map[row_id] = (piece_idx, placement)
            row_id += 1

    print(f"  DLX matrix: {row_id} rows x {len(all_columns)} columns")
    print("Solving...")

    # Solve
    raw_solutions = dlx.solve(find_all=find_all)

    # Convert solutions to piece -> placement mapping
    solutions = []
    for raw in raw_solutions:
        sol = {}
        for rid in raw:
            pidx, placement = row_map[rid]
            sol[pidx] = placement
        solutions.append(sol)

    print(f"Found {len(solutions)} solution(s).")
    return solutions
