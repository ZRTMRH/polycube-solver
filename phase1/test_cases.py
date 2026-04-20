"""
Test cases for the polycube solver.

Defines known puzzles and runs the solver to validate correctness.
"""

from .solver import solve


# ── Piece definitions ────────────────────────────────────────────────────────

# Monominoes (single cube)
MONOMINO = [(0, 0, 0)]

# Dominoes (2 cubes)
DOMINO = [(0, 0, 0), (1, 0, 0)]

# Tricubes
L_TRICUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]  # L-shape in xy-plane
I_TRICUBE = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]  # straight line

# Tetracubes (4 cubes)
I_TETRACUBE = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
L_TETRACUBE = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0)]
T_TETRACUBE = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0)]
S_TETRACUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]  # skew/S-shape
O_TETRACUBE = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]  # square
BRANCH_TETRACUBE = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)]  # T/branch in 3D
LEFT_SCREW = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]  # left screw tetracube
RIGHT_SCREW = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 1)]  # right screw (mirror)... wait

# Pentacubes would have 5 cubes — skip for now

# ── Soma Cube pieces (7 pieces, 27 cubes total → 3x3x3) ─────────────────────
# The Soma cube consists of all irregular polycubes with 3-4 unit cubes.
# Piece names follow standard convention (V, L, T, Z, S, A, B or 1-7).

SOMA_V = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]                      # V-tricube (piece 1)
SOMA_L = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0)]           # L-tetracube (piece 2)
SOMA_T = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0)]           # T-tetracube (piece 3)
SOMA_Z = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]           # Z/S-tetracube (piece 4)
SOMA_S = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]           # S-tetracube mirror...
# Actually Z and S are distinct in 2D but the same in 3D (rotatable into each other).
# Let's use the correct Soma pieces:

# Soma piece definitions (standard numbering):
# 1: V-shape (3 cubes, the only tricube)
# 2: L-tetracube
# 3: T-tetracube
# 4: Z-tetracube (S-shape)
# 5: Left screw (3D piece)
# 6: Right screw (3D piece, mirror of 5)
# 7: Branch/tripod (3D piece)

SOMA_PIECES = [
    # Piece 1: V-tricube (L-shape with 3 cubes)
    [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
    # Piece 2: L-tetracube
    [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0)],
    # Piece 3: T-tetracube
    [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0)],
    # Piece 4: Z-tetracube (skew)
    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)],
    # Piece 5: Left screw (uses 3rd dimension)
    [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)],
    # Piece 6: Right screw (mirror of 5, uses 3rd dimension)
    [(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)],
    # Piece 7: Branch/tripod (uses 3rd dimension)
    [(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)],
]


# ── Test functions ───────────────────────────────────────────────────────────

def test_2x2x2_monominoes():
    """8 monominoes should trivially fill a 2x2x2 cube."""
    print("=" * 60)
    print("TEST: 2x2x2 with 8 monominoes")
    pieces = [MONOMINO] * 8
    solutions = solve(pieces)
    assert len(solutions) >= 1, "Should have at least 1 solution"
    # Verify: all 8 cells covered exactly once
    sol = solutions[0]
    all_cells = set()
    for cells in sol.values():
        assert all_cells.isdisjoint(cells), "Overlap detected!"
        all_cells |= cells
    assert len(all_cells) == 8, f"Expected 8 cells, got {len(all_cells)}"
    print("PASSED\n")
    return solutions


def test_2x2x2_dominoes():
    """4 dominoes should fill a 2x2x2 cube."""
    print("=" * 60)
    print("TEST: 2x2x2 with 4 dominoes")
    pieces = [DOMINO] * 4
    solutions = solve(pieces)
    assert len(solutions) >= 1, "Should have at least 1 solution"
    sol = solutions[0]
    all_cells = set()
    for cells in sol.values():
        assert all_cells.isdisjoint(cells), "Overlap detected!"
        all_cells |= cells
    assert len(all_cells) == 8, f"Expected 8 cells, got {len(all_cells)}"
    print("PASSED\n")
    return solutions


def test_2x2x2_impossible():
    """Volume mismatch: 3 dominoes = 6 cubes ≠ 8. Should fail."""
    print("=" * 60)
    print("TEST: 2x2x2 impossible (3 dominoes, volume=6)")
    pieces = [DOMINO] * 3
    solutions = solve(pieces)
    assert len(solutions) == 0, "Should have no solutions"
    print("PASSED\n")
    return solutions


def test_2x2x2_l_tricube_plus_monominos():
    """1 L-tricube + 5 monominoes → 2x2x2 cube."""
    print("=" * 60)
    print("TEST: 2x2x2 with 1 L-tricube + 5 monominoes")
    pieces = [L_TRICUBE] + [MONOMINO] * 5
    solutions = solve(pieces)
    assert len(solutions) >= 1, "Should have at least 1 solution"
    print("PASSED\n")
    return solutions


def test_soma_cube():
    """The classic Soma cube: 7 pieces → 3x3x3. Known to have 240 solutions."""
    print("=" * 60)
    print("TEST: Soma Cube (7 pieces → 3x3x3)")
    total = sum(len(p) for p in SOMA_PIECES)
    print(f"  Total volume: {total}")
    assert total == 27, f"Soma pieces should total 27 cubes, got {total}"

    solutions = solve(SOMA_PIECES)
    assert len(solutions) >= 1, "Soma cube should have solutions"
    print(f"  Found {len(solutions)} solution(s)")

    # Verify first solution
    sol = solutions[0]
    all_cells = set()
    for cells in sol.values():
        assert all_cells.isdisjoint(cells), "Overlap detected!"
        all_cells |= cells
    assert len(all_cells) == 27, f"Expected 27 cells, got {len(all_cells)}"
    print("PASSED\n")
    return solutions


def test_soma_cube_all():
    """Find ALL Soma cube solutions. Should be 11520 (= 240 unique × 48 symmetries)."""
    print("=" * 60)
    print("TEST: Soma Cube — finding ALL solutions")
    solutions = solve(SOMA_PIECES, find_all=True)
    print(f"  Found {len(solutions)} solutions (expected 11520 = 240 × 48)")
    # 240 unique solutions × 48 cube symmetries (rotations + reflections)
    assert len(solutions) == 11520, f"Expected 11520 solutions, got {len(solutions)}"
    print("PASSED\n")
    return solutions


def verify_solution(solution, grid_size, pieces=None):
    """Verify that a solution correctly fills an NxNxN cube.

    Checks:
      1. No overlapping cells between pieces
      2. All cells in [0, grid_size)^3 are covered (no gaps)
      3. Every piece index in [0, len(pieces)) appears exactly once
      4. If `pieces` is provided, each placed shape matches the input
         piece up to rotation and translation

    Args:
        solution: dict mapping piece_idx -> frozenset/set of (x,y,z) cells
        grid_size: side length of target cube
        pieces: optional list of input pieces (each a list of (x,y,z) tuples).
            When provided, enables shape-match verification.

    Returns:
        True if the solution is valid
    """
    from .polycube import normalize, get_orientations

    expected_cells = set()
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                expected_cells.add((x, y, z))

    # Check piece index completeness: every index 0..n-1 must appear exactly once
    if pieces is not None:
        expected_indices = set(range(len(pieces)))
        actual_indices = set(solution.keys())
        if actual_indices != expected_indices:
            missing = expected_indices - actual_indices
            extra = actual_indices - expected_indices
            print(f"  ERROR: Piece index mismatch — missing: {missing}, extra: {extra}")
            return False

    actual_cells = set()
    for piece_idx, cells in solution.items():
        cells_set = frozenset(cells) if not isinstance(cells, frozenset) else cells

        # Check overlap
        overlap = actual_cells & cells_set
        if overlap:
            print(f"  ERROR: Piece {piece_idx} overlaps at {overlap}")
            return False
        actual_cells |= cells_set

        # Check bounds
        for x, y, z in cells_set:
            if not (0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size):
                print(f"  ERROR: Piece {piece_idx} cell ({x},{y},{z}) out of bounds")
                return False

        # Check shape matches input piece (up to rotation + translation)
        if pieces is not None:
            if piece_idx < 0 or piece_idx >= len(pieces):
                print(f"  ERROR: Piece index {piece_idx} out of range [0, {len(pieces)})")
                return False
            placed_norm = normalize(cells_set)
            valid_orientations = set(get_orientations(pieces[piece_idx]))
            if placed_norm not in valid_orientations:
                print(f"  ERROR: Piece {piece_idx} placement doesn't match any rotation of input shape")
                return False

    # Check full coverage
    if actual_cells != expected_cells:
        missing = expected_cells - actual_cells
        extra = actual_cells - expected_cells
        if missing:
            print(f"  ERROR: Missing {len(missing)} cells")
        if extra:
            print(f"  ERROR: {len(extra)} extra cells outside grid")
        return False

    return True


if __name__ == "__main__":
    test_2x2x2_monominoes()
    test_2x2x2_dominoes()
    test_2x2x2_impossible()
    test_2x2x2_l_tricube_plus_monominos()
    test_soma_cube()

    print("=" * 60)
    print("ALL BASIC TESTS PASSED")
    print()

    # Uncomment to find all Soma solutions (takes a few seconds):
    # test_soma_cube_all()
