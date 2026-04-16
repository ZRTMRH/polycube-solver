"""
Unbiased constructive puzzle generator for 3D polycube packing.

Partitions an NxNxN cube into connected polycube pieces of size 3-5.

Algorithm:
  1. Pick a surface cube (a 1x1x1 cell with at least one missing neighbor).
  2. Enumerate ALL connected pieces of size 3-5 that contain that cube
     and fit within the remaining region.
  3. Randomly pick one whose removal keeps the remaining region connected.
  4. Remove it and repeat.

This is unbiased: every valid piece containing the chosen cube is equally
likely to be selected. No shape scoring, no growth heuristic.

Author: STA 561 Final Project
"""

import random
import sys
import time
from typing import List, Optional, Set, Tuple, Dict

Coord = Tuple[int, int, int]
Piece = List[Coord]

DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _neighbors(cell: Coord, N: int) -> List[Coord]:
    """6-connected neighbors within an NxNxN grid."""
    x, y, z = cell
    out = []
    if x > 0:     out.append((x - 1, y, z))
    if x < N - 1: out.append((x + 1, y, z))
    if y > 0:     out.append((x, y - 1, z))
    if y < N - 1: out.append((x, y + 1, z))
    if z > 0:     out.append((x, y, z - 1))
    if z < N - 1: out.append((x, y, z + 1))
    return out


def _is_connected(cells: Set[Coord], N: int) -> bool:
    """Check if all cells form a single connected component."""
    if len(cells) <= 1:
        return True
    start = next(iter(cells))
    visited = set()
    stack = [start]
    while stack:
        c = stack.pop()
        if c in visited:
            continue
        visited.add(c)
        for nb in _neighbors(c, N):
            if nb in cells and nb not in visited:
                stack.append(nb)
    return len(visited) == len(cells)


def _remaining_volume_feasible(rem: int) -> bool:
    """Check if rem can be partitioned into sizes from {3, 4, 5}."""
    return rem == 0 or rem >= 3


def _removal_ok(piece_set: Set[Coord], remaining: Set[Coord], N: int) -> bool:
    """Check if removing piece_set from remaining leaves a connected region.

    Uses boundary BFS: only checks that all neighbors of the removed piece
    can still reach each other.
    """
    after = remaining - piece_set
    if not after:
        return True

    # Find boundary cells: cells in `after` adjacent to the removed piece
    boundary = set()
    for cell in piece_set:
        for nb in _neighbors(cell, N):
            if nb in after:
                boundary.add(nb)

    if not boundary:
        return _is_connected(after, N)

    if len(boundary) == 1:
        return True

    # BFS from one boundary cell until all boundary cells are found
    start = next(iter(boundary))
    found = {start}
    target = len(boundary)
    visited = {start}
    stack = [start]
    while stack and len(found) < target:
        c = stack.pop()
        for nb in _neighbors(c, N):
            if nb in after and nb not in visited:
                visited.add(nb)
                stack.append(nb)
                if nb in boundary:
                    found.add(nb)

    return len(found) == target


# ---------------------------------------------------------------------------
# Piece enumeration: all connected pieces of size 3-5 containing a root cell
# ---------------------------------------------------------------------------

def _enumerate_pieces_at(root: Coord, remaining: Set[Coord],
                         N: int) -> Dict[int, List[frozenset]]:
    """Enumerate all connected pieces of size 3, 4, 5 containing root.

    Uses BFS-style level expansion:
      level 1: {root}
      level k: extend each level-(k-1) piece by one neighbor in remaining

    Frozensets auto-deduplicate identical pieces reached via different orderings.

    Returns: {size: [frozenset, ...]}
    """
    current = {frozenset([root])}
    result = {}

    for sz in range(2, 6):  # grow to sizes 2, 3, 4, 5
        nxt = set()
        for piece in current:
            for cell in piece:
                for nb in _neighbors(cell, N):
                    if nb in remaining and nb not in piece:
                        nxt.add(piece | {nb})
        current = nxt
        if sz >= 3:
            result[sz] = list(current)

    return result


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def build_robust_constructive_case(
    grid_size: int, seed: int, max_restarts: int = 200
) -> Optional[List[Piece]]:
    """Generate a valid 3D polycube packing instance via greedy decomposition.

    Partitions an N x N x N cube into pieces of size 3-5, each a connected
    polycube. The partition itself IS the solution, so it is always solvable.

    Algorithm: pick a surface cube, enumerate all valid pieces containing it,
    randomly select one that doesn't disconnect the remaining region.
    No shape bias — every valid piece is equally likely.

    Args:
        grid_size: side length N of the target cube
        seed: random seed for reproducibility
        max_restarts: max full-cube restart attempts

    Returns:
        List of pieces (each a list of (x, y, z) tuples), or None if
        generation fails after all retries.
    """
    for restart in range(max_restarts):
        rng = random.Random(seed * 1000 + restart)
        remaining = set(
            (x, y, z)
            for x in range(grid_size)
            for y in range(grid_size)
            for z in range(grid_size)
        )

        # Maintain surface set incrementally
        surface_set = set()
        for c in remaining:
            for d in DIRS:
                nb = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if nb not in remaining:
                    surface_set.add(c)
                    break

        pieces: List[Piece] = []
        failed = False

        while remaining:
            vol_left = len(remaining)
            if vol_left < 3:
                failed = True
                break

            # Determine valid piece sizes
            valid_sizes = [
                s for s in [3, 4, 5]
                if s <= vol_left and _remaining_volume_feasible(vol_left - s)
            ]
            if not valid_sizes:
                failed = True
                break

            # Pick a surface cube to build a piece around.
            # Shuffle surface cells so different restarts try different orders.
            surface_list = list(surface_set)
            rng.shuffle(surface_list)

            piece_cells = None
            for root in surface_list:
                # Enumerate all pieces of valid sizes containing this root
                all_pieces = _enumerate_pieces_at(root, remaining, grid_size)

                # Collect candidates of valid sizes, shuffled
                candidates = []
                for sz in valid_sizes:
                    candidates.extend(all_pieces.get(sz, []))
                rng.shuffle(candidates)

                # Try each candidate — pick the first that doesn't disconnect
                for cand in candidates:
                    cand_set = set(cand)
                    if _removal_ok(cand_set, remaining, grid_size):
                        piece_cells = cand_set
                        break

                if piece_cells is not None:
                    break

            if piece_cells is None:
                failed = True
                break

            remaining -= piece_cells
            # Update surface set
            surface_set -= piece_cells
            for cell in piece_cells:
                for nb in _neighbors(cell, grid_size):
                    if nb in remaining:
                        surface_set.add(nb)
            pieces.append(list(piece_cells))

        if not failed and not remaining:
            return pieces

    return None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_partition(pieces: List[Piece], grid_size: int) -> bool:
    """Verify that pieces form a valid partition of the NxNxN cube."""
    N = grid_size
    expected = {
        (x, y, z)
        for x in range(N)
        for y in range(N)
        for z in range(N)
    }

    all_cells = set()
    for i, piece in enumerate(pieces):
        piece_set = set(piece)
        if not _is_connected(piece_set, N):
            print(f"  FAIL: piece {i} is not connected")
            return False
        if len(piece) < 3 or len(piece) > 5:
            print(f"  FAIL: piece {i} has size {len(piece)} (must be 3-5)")
            return False
        overlap = all_cells & piece_set
        if overlap:
            print(f"  FAIL: piece {i} overlaps at {overlap}")
            return False
        for x, y, z in piece:
            if not (0 <= x < N and 0 <= y < N and 0 <= z < N):
                print(f"  FAIL: piece {i} cell ({x},{y},{z}) out of bounds")
                return False
        all_cells |= piece_set

    if all_cells != expected:
        missing = expected - all_cells
        extra = all_cells - expected
        print(f"  FAIL: coverage mismatch. Missing={len(missing)}, Extra={len(extra)}")
        return False

    return True


def piece_dimensionality(piece: Piece) -> int:
    """Count how many axes a piece spans (1=rod, 2=slab, 3=genuinely 3D)."""
    xs = set(c[0] for c in piece)
    ys = set(c[1] for c in piece)
    zs = set(c[2] for c in piece)
    return (len(xs) > 1) + (len(ys) > 1) + (len(zs) > 1)


# ---------------------------------------------------------------------------
# Main: test at various grid sizes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_sizes = [3, 4, 5, 6, 7, 9, 10, 11, 12]
    num_seeds = 3

    for N in test_sizes:
        print(f"\n{'=' * 60}")
        print(f"Grid size {N}x{N}x{N} (volume={N**3})")
        print(f"{'=' * 60}")
        sys.stdout.flush()

        for s in range(num_seeds):
            seed = N * 1000 + s * 7 + 42
            t0 = time.time()
            pieces = build_robust_constructive_case(N, seed)
            elapsed = time.time() - t0

            if pieces is None:
                print(f"  seed={seed}: FAILED after {elapsed:.2f}s")
                sys.stdout.flush()
                continue

            valid = validate_partition(pieces, N)
            dims = [piece_dimensionality(p) for p in pieces]
            dim_counts = {1: dims.count(1), 2: dims.count(2), 3: dims.count(3)}
            pct_3d = 100.0 * dim_counts[3] / len(pieces) if pieces else 0

            status = "PASS" if valid else "FAIL"
            print(
                f"  seed={seed}: {status} | {len(pieces):3d} pcs | "
                f"{elapsed:6.3f}s | 1D/2D/3D="
                f"{dim_counts[1]}/{dim_counts[2]}/{dim_counts[3]} "
                f"({pct_3d:.0f}% 3D)"
            )
            sys.stdout.flush()
