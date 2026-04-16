"""
Robust constructive puzzle generator for 3D polycube packing.

Reliably partitions an NxNxN cube into connected polycube pieces of
size 3-5 for grid sizes 3 through 9+.

Algorithm: "peel from the outside"
  - Maintain a set of remaining cells and a neighbor-count map.
  - Always grow pieces starting from "leaf" cells (cells with few
    neighbors in the remaining set). This naturally peels pieces
    from the surface inward, avoiding the fragmentation problem
    that plagues random BFS growth.
  - Only check connectivity of the remaining region after each
    complete piece is removed (not at every cell addition).
  - If a piece removal disconnects the remaining region, reject
    that piece and try a different growth.

Author: robust_generator for STA 561 Final Project
"""

import random
import sys
import time
from collections import deque
from typing import List, Optional, Set, Tuple, Dict

Coord = Tuple[int, int, int]
Piece = List[Coord]


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


def _connected_components(cells: Set[Coord], N: int) -> List[Set[Coord]]:
    """Find connected components via DFS."""
    remaining = set(cells)
    components = []
    while remaining:
        seed = next(iter(remaining))
        comp = set()
        stack = [seed]
        while stack:
            c = stack.pop()
            if c in comp:
                continue
            comp.add(c)
            for nb in _neighbors(c, N):
                if nb in remaining and nb not in comp:
                    stack.append(nb)
        components.append(comp)
        remaining -= comp
    return components


def _remaining_volume_feasible(rem: int) -> bool:
    """Check if rem can be partitioned into sizes from {3, 4, 5}."""
    # All integers >= 3 are feasible (3=3, 4=4, 5=5, 6=3+3, 7=3+4, ...)
    return rem == 0 or rem >= 3


def _sample_piece_sizes(rng: random.Random, total: int) -> Optional[List[int]]:
    """Sample piece sizes in {3,4,5} summing to total."""
    if total < 3 and total != 0:
        return None
    sizes = []
    remain = total
    while remain > 0:
        choices = [s for s in (3, 4, 5) if _remaining_volume_feasible(remain - s)]
        if not choices:
            return None
        s = rng.choice(choices)
        sizes.append(s)
        remain -= s
    return sizes if remain == 0 else None


# ---------------------------------------------------------------------------
# Neighbor-count based efficient growing
# ---------------------------------------------------------------------------

def _build_nbcount(remaining: Set[Coord], N: int) -> Dict[Coord, int]:
    """For each cell in remaining, count how many of its 6-neighbors
    are also in remaining."""
    nbcount = {}
    for cell in remaining:
        cnt = 0
        for nb in _neighbors(cell, N):
            if nb in remaining:
                cnt += 1
        nbcount[cell] = cnt
    return nbcount


def _grow_piece_peeling(rng: random.Random, remaining: Set[Coord],
                        N: int, target_size: int,
                        nbcount: Dict[Coord, int]) -> Optional[Set[Coord]]:
    """Grow a piece by starting from a low-neighbor-count cell (surface cell)
    and growing along the surface. This "peels" pieces from the outside,
    preserving connectivity of the interior.

    Returns the piece cells, or None if growth failed.
    """
    if len(remaining) < target_size:
        return None
    if len(remaining) == target_size:
        if _is_connected(remaining, N):
            return set(remaining)
        return None

    # Find cells with lowest neighbor count (surface cells)
    min_nb = min(nbcount[c] for c in remaining)
    surface = [c for c in remaining if nbcount[c] <= min_nb + 1]
    rng.shuffle(surface)

    for seed_cell in surface[:15]:
        piece = {seed_cell}

        # BFS frontier: neighbors of piece that are in remaining
        frontier = []
        for nb in _neighbors(seed_cell, N):
            if nb in remaining and nb not in piece:
                frontier.append(nb)

        while len(piece) < target_size:
            if not frontier:
                break

            # Score frontier cells: prefer low nbcount (surface cells)
            # and cells that extend dimensionality
            scored = []
            for fc in frontier:
                if fc not in remaining or fc in piece:
                    continue
                # Base score: prefer surface cells (low neighbor count)
                nb_score = -nbcount.get(fc, 0)

                # Dimensionality bonus
                xs = set(p[0] for p in piece) | {fc[0]}
                ys = set(p[1] for p in piece) | {fc[1]}
                zs = set(p[2] for p in piece) | {fc[2]}
                dim_bonus = (len(xs) > 1) + (len(ys) > 1) + (len(zs) > 1)

                scored.append((fc, nb_score + dim_bonus * 2))

            if not scored:
                break

            # Pick from top candidates with randomness
            scored.sort(key=lambda x: -x[1])
            top_score = scored[0][1]
            best = [fc for fc, s in scored if s >= top_score - 1]
            nxt = rng.choice(best)

            piece.add(nxt)

            # Rebuild frontier (simple: just neighbors of piece in remaining)
            new_frontier = set()
            for pc in piece:
                for nb in _neighbors(pc, N):
                    if nb in remaining and nb not in piece:
                        new_frontier.add(nb)
            frontier = list(new_frontier)

        if len(piece) == target_size:
            return piece

    return None


def _piece_removal_ok(piece: Set[Coord], remaining: Set[Coord],
                      N: int) -> bool:
    """Check if removing piece from remaining leaves feasible components."""
    after = remaining - piece
    if not after:
        return True
    comps = _connected_components(after, N)
    for comp in comps:
        if not _remaining_volume_feasible(len(comp)):
            return False
    return True


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def build_robust_constructive_case(grid_size: int, seed: int) -> List[Piece]:
    """Generate a guaranteed-solvable 3D polycube packing instance.

    Partitions an N x N x N cube into pieces of size 3-5, each a connected
    polycube. The partition itself IS the solution, so it is always solvable.

    Args:
        grid_size: side length N of the target cube
        seed: random seed for reproducibility

    Returns:
        List of pieces, each a list of (x, y, z) tuples.

    Raises:
        RuntimeError if generation fails after many retries.
    """
    rng = random.Random(seed)
    total = grid_size ** 3
    max_restarts = 2000

    for restart in range(max_restarts):
        sizes = _sample_piece_sizes(rng, total)
        if sizes is None:
            continue

        rng.shuffle(sizes)

        remaining = {
            (x, y, z)
            for x in range(grid_size)
            for y in range(grid_size)
            for z in range(grid_size)
        }

        nbcount = _build_nbcount(remaining, grid_size)

        pieces: List[Piece] = []
        ok = True

        for i, sz in enumerate(sizes):
            # Try multiple times to grow a piece that doesn't fragment
            piece_cells = None
            for attempt in range(10):
                candidate = _grow_piece_peeling(
                    rng, remaining, grid_size, sz, nbcount
                )
                if candidate is None:
                    break
                # Check that removal doesn't create tiny components
                if _piece_removal_ok(candidate, remaining, grid_size):
                    piece_cells = candidate
                    break
                # Otherwise try again with different randomness

            if piece_cells is None:
                ok = False
                break

            # Update remaining and nbcount
            remaining -= piece_cells
            # Rebuild nbcount for remaining cells near the removed piece
            affected = set()
            for cell in piece_cells:
                for nb in _neighbors(cell, grid_size):
                    if nb in remaining:
                        affected.add(nb)
            for cell in piece_cells:
                nbcount.pop(cell, None)
            for cell in affected:
                cnt = 0
                for nb in _neighbors(cell, grid_size):
                    if nb in remaining:
                        cnt += 1
                nbcount[cell] = cnt

            pieces.append(list(piece_cells))

        if ok and not remaining:
            return pieces

    raise RuntimeError(
        f"Failed to construct case for grid_size={grid_size} after "
        f"{max_restarts} restarts."
    )


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
    test_sizes = [3, 4, 5, 6, 7, 9]
    num_seeds = 3

    for N in test_sizes:
        print(f"\n{'='*60}")
        print(f"Grid size {N}x{N}x{N} (volume={N**3})")
        print(f"{'='*60}")
        sys.stdout.flush()

        for s in range(num_seeds):
            seed = N * 1000 + s * 7 + 42
            t0 = time.time()
            try:
                pieces = build_robust_constructive_case(N, seed)
                elapsed = time.time() - t0

                valid = validate_partition(pieces, N)
                dims = [piece_dimensionality(p) for p in pieces]
                dim_counts = {1: dims.count(1), 2: dims.count(2), 3: dims.count(3)}
                pct_3d = 100.0 * dim_counts[3] / len(pieces) if pieces else 0

                status = "PASS" if valid else "FAIL"
                print(f"  seed={seed}: {status} | {len(pieces):3d} pcs | "
                      f"{elapsed:6.2f}s | 1D/2D/3D="
                      f"{dim_counts[1]}/{dim_counts[2]}/{dim_counts[3]} "
                      f"({pct_3d:.0f}% 3D)")
            except RuntimeError as e:
                elapsed = time.time() - t0
                print(f"  seed={seed}: ERROR after {elapsed:.2f}s - {e}")
            sys.stdout.flush()

    # --- DLX solver verification for small sizes ---
    print(f"\n{'='*60}")
    print("DLX solver verification (grid sizes 3-5)")
    print(f"{'='*60}")
    sys.stdout.flush()

    try:
        from solver import solve
        for N in [3, 4, 5]:
            seed = N * 1000 + 99
            pieces = build_robust_constructive_case(N, seed)
            valid = validate_partition(pieces, N)
            if not valid:
                print(f"  N={N}: partition invalid, skipping DLX")
                continue
            print(f"\n  N={N}: running DLX solver on {len(pieces)} pieces...")
            sys.stdout.flush()
            solutions = solve(pieces, grid_size=N)
            if solutions:
                print(f"  N={N}: DLX found {len(solutions)} solution(s) -- PASS")
            else:
                print(f"  N={N}: DLX found NO solutions -- FAIL")
            sys.stdout.flush()
    except ImportError as e:
        print(f"  Could not import solver: {e}")
    except Exception as e:
        print(f"  Solver error: {e}")
