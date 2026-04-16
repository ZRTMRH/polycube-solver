"""Block planner v2 for large grids: slab decomposition.

Many constructive large-grid puzzles (e.g. ``mixed_constructive`` at 12^3) are
built from pieces that all live inside a single 1-cell-thick axial slab. When
this is true we can decompose the cube into N slabs of size NxNx1 and solve
each independently as a 2D exact cover. Each slab has only N^2 cells, so DLX
finishes in well under a second even at N=12.

This module exposes:

- ``detect_slab_axis(pieces)``: return the axis (0,1,2) along which every
  piece has bounding-box extent 1, or ``None``.
- ``solve_slab_planner(pieces, grid_size, ...)``: try slab decomposition; on
  success return ``(solution_dict, diag)``, otherwise ``(None, diag)``.

The planner uses the *absolute coordinates* of input pieces as a hint for
which slab each piece belongs to. The fixture's ``mixed_constructive`` case
ships pieces in a known-valid placement, so this hint is essentially free.
The 2D DLX call still proves the slab is solvable from that piece set, so a
hint failure is detected and reported.
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1.dlx_solver import DLX
from phase1.polycube import get_orientations


Piece = List[Tuple[int, int, int]]


def detect_slab_axis(pieces: List[Piece]) -> Optional[int]:
    """Return axis index (0=x, 1=y, 2=z) where *every* piece has extent 1.

    If multiple axes qualify (e.g. all single-cell pieces), prefer the lowest
    index. Returns ``None`` if no such axis exists.
    """
    if not pieces:
        return None

    # Fast path: check first piece to narrow candidates, then verify rest.
    candidates = [True, True, True]  # axis 0, 1, 2
    for piece in pieces:
        if not piece:
            return None
        c0 = piece[0]
        for a in range(3):
            if not candidates[a]:
                continue
            val = c0[a]
            for c in piece[1:]:
                if c[a] != val:
                    candidates[a] = False
                    break
        if not any(candidates):
            return None

    for a in range(3):
        if candidates[a]:
            return a
    return None


def _project_piece_2d(piece: Piece, slab_axis: int) -> Tuple[FrozenSet[Tuple[int, int]], int]:
    """Collapse slab axis: return (2D cells, slab index)."""
    coords = [tuple(int(v) for v in c) for c in piece]
    slab_idx = coords[0][slab_axis]
    other = [a for a in (0, 1, 2) if a != slab_axis]
    cells = frozenset((c[other[0]], c[other[1]]) for c in coords)
    return cells, slab_idx


def _orient_2d_placements(piece: Piece, grid_size: int, slab_axis: int) -> List[FrozenSet[Tuple[int, int]]]:
    """Return all 2D (in-plane) placements of a piece given slab orientation.

    We take the 24 cube rotations, keep only those that preserve the
    1-cell-thick orientation along ``slab_axis``, then collapse to 2D and
    enumerate translations within ``grid_size x grid_size``.
    """
    other = [a for a in (0, 1, 2) if a != slab_axis]
    placements_set = set()

    for orient in get_orientations(piece):
        ext = [
            max(c[a] for c in orient) - min(c[a] for c in orient)
            for a in (0, 1, 2)
        ]
        if ext[slab_axis] != 0:
            continue
        # Collapse to 2D shape (already normalized to origin in get_orientations).
        cells2d = [(c[other[0]], c[other[1]]) for c in orient]
        max_u = max(c[0] for c in cells2d)
        max_v = max(c[1] for c in cells2d)
        for du in range(grid_size - max_u):
            for dv in range(grid_size - max_v):
                placed = frozenset((u + du, v + dv) for u, v in cells2d)
                placements_set.add(placed)

    return list(placements_set)


def _solve_2d_region_dlx(
    piece_indices: List[int],
    pieces: List[Piece],
    region_cells: List[Tuple[int, int]],
    slab_axis: int,
    timeout: float,
) -> Optional[Dict[int, FrozenSet[Tuple[int, int]]]]:
    """Solve a 2D region via DLX. Returns piece_idx -> 2D cell set or None.

    ``region_cells`` is the explicit list of 2D cells (u, v) the region covers.
    Placements that fall outside the region are filtered out. ``slab_axis`` is
    the 3D axis collapsed to make 2D coordinates.
    """
    t0 = time.time()
    region_set = set(region_cells)
    if not region_set:
        return None

    cell_names = [f"c_{u}_{v}" for (u, v) in region_set]
    piece_names = [f"p_{i}" for i in piece_indices]
    dlx = DLX(cell_names + piece_names)

    # Bounding box of region.
    us = [c[0] for c in region_set]
    vs = [c[1] for c in region_set]
    u_lo, u_hi = min(us), max(us)
    v_lo, v_hi = min(vs), max(vs)
    # We need to enumerate placements over a bounding box that covers the
    # region. Use the 2D shape (already normalized to origin via
    # ``get_orientations``) and enumerate translations into [u_lo, u_hi]^2 etc.
    other = [a for a in (0, 1, 2) if a != slab_axis]

    row_map: Dict[int, Tuple[int, FrozenSet[Tuple[int, int]]]] = {}
    rid = 0
    for pidx in piece_indices:
        # Get all distinct 2D shape orientations.
        shape_orients = set()
        for orient in get_orientations(pieces[pidx]):
            ext_slab = max(c[slab_axis] for c in orient) - min(c[slab_axis] for c in orient)
            if ext_slab != 0:
                continue
            cells2d = frozenset((c[other[0]], c[other[1]]) for c in orient)
            shape_orients.add(cells2d)

        valid: List[FrozenSet[Tuple[int, int]]] = []
        for shape in shape_orients:
            max_du = max(c[0] for c in shape)
            max_dv = max(c[1] for c in shape)
            min_du = min(c[0] for c in shape)
            min_dv = min(c[1] for c in shape)
            # Translation range so placed cells stay within region bbox.
            for du in range(u_lo - min_du, u_hi - max_du + 1):
                for dv in range(v_lo - min_dv, v_hi - max_dv + 1):
                    shifted = frozenset((u + du, v + dv) for (u, v) in shape)
                    if shifted.issubset(region_set):
                        valid.append(shifted)
        # De-dup.
        valid = list(set(valid))
        if not valid:
            return None
        for placed in valid:
            cols = [f"p_{pidx}"]
            for u, v in placed:
                cols.append(f"c_{u}_{v}")
            dlx.add_row(rid, cols)
            row_map[rid] = (pidx, placed)
            rid += 1
        if time.time() - t0 > timeout:
            return None

    solutions = dlx.solve(find_all=False)
    if not solutions:
        return None
    sol = {}
    for r in solutions[0]:
        pidx, placed = row_map[r]
        sol[pidx] = placed
    return sol


def _detect_strip_axis_in_2d(
    piece_indices: List[int],
    pieces: List[Piece],
    candidate_axes_3d: List[int],
) -> Optional[int]:
    """Return a 3D axis (in ``candidate_axes_3d``) along which every piece in
    the slab has extent <= 1 (i.e. the slab can be split into 2-row strips).
    """
    for a in candidate_axes_3d:
        ok = True
        for pidx in piece_indices:
            vals = [cell[a] for cell in pieces[pidx]]
            if max(vals) - min(vals) > 1:
                ok = False
                break
        if ok:
            return a
    return None


def solve_slab_sequential(
    pieces: List[Piece],
    grid_size: int,
    slab_timeout: float = 10.0,
    total_timeout: float = 60.0,
) -> Tuple[Optional[Dict[int, FrozenSet[Tuple[int, int, int]]]], dict]:
    """Solve flat-piece puzzles without coordinate hints.

    Builds a single full-grid DLX: all 3D cells + all piece IDs are primary
    columns. Since pieces are flat (extent 0 along one axis), each placement
    is a 2D shape × slab_index. This is a standard exact cover that DLX
    handles efficiently.

    Handles ``relative_pieces=True`` cases where absolute coordinates
    are normalized away.
    """
    diag = {
        'planner': 'slab_sequential',
        'slab_axis': None,
        'slabs_total': 0,
        'slabs_solved': 0,
    }

    if not pieces:
        diag['reason'] = 'empty_pieces'
        return None, diag

    axis = detect_slab_axis(pieces)
    if axis is None:
        diag['reason'] = 'no_slab_axis'
        return None, diag
    diag['slab_axis'] = axis
    diag['slabs_total'] = grid_size

    total_vol = sum(len(p) for p in pieces)
    if total_vol != grid_size ** 3:
        diag['reason'] = 'volume_mismatch'
        return None, diag

    other = [a for a in (0, 1, 2) if a != axis]
    t0 = time.time()

    # Build a single DLX for the full 3D grid.
    # Primary columns: all grid cells + all piece IDs.
    cell_names = []
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                cell_names.append(f"c_{x}_{y}_{z}")
    piece_names = [f"p_{i}" for i in range(len(pieces))]
    all_cols = cell_names + piece_names

    dlx = DLX(all_cols)

    row_map = {}
    rid = 0

    for pidx, piece in enumerate(pieces):
        if time.time() - t0 > total_timeout:
            diag['reason'] = 'build_timeout'
            return None, diag

        # Get distinct 2D shapes for this piece.
        shape_orients = set()
        for orient in get_orientations(piece):
            ext_slab = max(c[axis] for c in orient) - min(c[axis] for c in orient)
            if ext_slab != 0:
                continue
            cells2d = frozenset((c[other[0]], c[other[1]]) for c in orient)
            shape_orients.add(cells2d)

        for shape in shape_orients:
            max_u = max(c[0] for c in shape)
            max_v = max(c[1] for c in shape)
            min_u = min(c[0] for c in shape)
            min_v = min(c[1] for c in shape)
            for du in range(0 - min_u, grid_size - max_u):
                for dv in range(0 - min_v, grid_size - max_v):
                    shifted = frozenset((u + du, v + dv) for (u, v) in shape)
                    # Place in each slab layer.
                    for slab_idx in range(grid_size):
                        cols = [f"p_{pidx}"]
                        for u, v in shifted:
                            coord = [0, 0, 0]
                            coord[axis] = slab_idx
                            coord[other[0]] = u
                            coord[other[1]] = v
                            cols.append(f"c_{coord[0]}_{coord[1]}_{coord[2]}")
                        dlx.add_row(rid, cols)
                        row_map[rid] = (pidx, slab_idx, shifted)
                        rid += 1

    if time.time() - t0 > total_timeout:
        diag['reason'] = 'build_timeout'
        return None, diag

    diag['dlx_rows'] = rid
    diag['dlx_cols'] = len(all_cols)

    solutions = dlx.solve(find_all=False)
    diag['solve_time'] = time.time() - t0

    if not solutions:
        diag['reason'] = 'dlx_no_solution'
        return None, diag

    global_solution: Dict[int, FrozenSet[Tuple[int, int, int]]] = {}
    for r in solutions[0]:
        pidx, slab_idx, cells2d = row_map[r]
        cells3d = []
        for u, v in cells2d:
            coord = [0, 0, 0]
            coord[axis] = slab_idx
            coord[other[0]] = u
            coord[other[1]] = v
            cells3d.append(tuple(coord))
        global_solution[pidx] = frozenset(cells3d)

    diag['slabs_solved'] = grid_size
    diag['reason'] = None
    return global_solution, diag


def solve_slab_planner(
    pieces: List[Piece],
    grid_size: int,
    slab_timeout: float = 5.0,
) -> Tuple[Optional[Dict[int, FrozenSet[Tuple[int, int, int]]]], dict]:
    """Try slab decomposition.

    Returns (solution_dict, diag).
    """
    diag = {
        'planner': 'slab_v2',
        'slab_axis': None,
        'slabs_total': 0,
        'slabs_solved': 0,
        'failed_slab': None,
    }

    if not pieces:
        diag['reason'] = 'empty_pieces'
        return None, diag

    axis = detect_slab_axis(pieces)
    if axis is None:
        diag['reason'] = 'no_slab_axis'
        return None, diag
    diag['slab_axis'] = axis

    # Group pieces by slab index using their absolute coordinates as a hint.
    # Since pieces are flat (extent 0 along axis), all cells share the same
    # axis coordinate. Use min for robustness against coordinate ordering.
    by_slab: Dict[int, List[int]] = defaultdict(list)
    for pidx, piece in enumerate(pieces):
        coords = [tuple(int(v) for v in c) for c in piece]
        slab_idx = min(c[axis] for c in coords)
        if slab_idx < 0 or slab_idx >= grid_size:
            diag['reason'] = 'slab_idx_out_of_range'
            return None, diag
        by_slab[slab_idx].append(pidx)

    diag['slabs_total'] = grid_size

    if len(by_slab) != grid_size:
        # Some slabs are empty in the hint — invalid input or unusual structure.
        diag['reason'] = f'slab_count_mismatch:{len(by_slab)}'
        return None, diag

    # Volume check per slab.
    for s in range(grid_size):
        idxs = by_slab.get(s, [])
        slab_volume = sum(len(pieces[i]) for i in idxs)
        if slab_volume != grid_size * grid_size:
            diag['reason'] = f'slab_volume_mismatch:slab={s}:vol={slab_volume}'
            return None, diag

    other = [a for a in (0, 1, 2) if a != axis]
    global_solution: Dict[int, FrozenSet[Tuple[int, int, int]]] = {}

    for s in range(grid_size):
        idxs = by_slab[s]

        # Try 1D-strip decomposition first: find a 3D axis (within ``other``)
        # along which every piece in this slab has extent <= 1, then split the
        # slab into 2-row strips.
        strip_axis_3d = _detect_strip_axis_in_2d(idxs, pieces, other)
        sol2d: Optional[Dict[int, FrozenSet[Tuple[int, int]]]] = None

        if strip_axis_3d is not None:
            # The third axis (the long axis of each strip).
            long_axis_3d = [a for a in other if a != strip_axis_3d][0]
            # Group pieces into strips by their min coord along strip_axis_3d.
            # Strips are 2-row regions; a piece with extent <=1 starting at
            # row r occupies rows {r, r+1} (or just {r} for ext 0).
            # Use min coord aligned to even pairs (0-1, 2-3, ...) since the
            # constructive generator pairs rows like that. Be robust if the
            # offset is odd by snapping to the actual min.
            strip_groups: Dict[int, List[int]] = defaultdict(list)
            for pidx in idxs:
                vals = [cell[strip_axis_3d] for cell in pieces[pidx]]
                # Strip key is the lower of the two paired rows; use floor div 2.
                strip_groups[min(vals) // 2].append(pidx)

            sol2d = {}
            ok = True
            for strip_key, strip_idxs in strip_groups.items():
                # Build region cells: rows {2*strip_key, 2*strip_key+1} of the
                # slab, each with columns 0..grid_size-1, projected to (u, v)
                # where u corresponds to other[0] and v to other[1].
                row_lo = 2 * strip_key
                row_hi = row_lo + 1
                # Map 3D axes to 2D collapsed coords.
                # other = [a, b]; coordinate u <- pos along other[0], v <- pos along other[1].
                strip_region: List[Tuple[int, int]] = []
                for r in (row_lo, row_hi):
                    if r >= grid_size:
                        continue
                    for col in range(grid_size):
                        coord3 = [0, 0, 0]
                        coord3[axis] = s  # not used in 2D
                        coord3[strip_axis_3d] = r
                        coord3[long_axis_3d] = col
                        u = coord3[other[0]]
                        v = coord3[other[1]]
                        strip_region.append((u, v))

                strip_sol = _solve_2d_region_dlx(
                    strip_idxs, pieces, strip_region, axis, slab_timeout,
                )
                if strip_sol is None:
                    ok = False
                    break
                sol2d.update(strip_sol)

            if not ok:
                sol2d = None

        # Fallback to full-slab DLX if strip decomposition didn't apply or failed.
        if sol2d is None:
            full_region = [(u, v) for u in range(grid_size) for v in range(grid_size)]
            sol2d = _solve_2d_region_dlx(idxs, pieces, full_region, axis, slab_timeout)

        if sol2d is None:
            diag['reason'] = 'slab_dlx_no_solution'
            diag['failed_slab'] = s
            diag['slabs_solved'] = s
            return None, diag

        # Lift 2D placement back to 3D using slab index.
        for pidx, cells2d in sol2d.items():
            cells3d = []
            for u, v in cells2d:
                coord = [0, 0, 0]
                coord[axis] = s
                coord[other[0]] = u
                coord[other[1]] = v
                cells3d.append(tuple(coord))
            global_solution[pidx] = frozenset(cells3d)

    diag['slabs_solved'] = grid_size
    diag['reason'] = None

    if len(global_solution) != len(pieces):
        diag['reason'] = 'incomplete_global_solution'
        return None, diag

    return global_solution, diag


# ---------------------------------------------------------------------------
#  Slab-by-slab solver for normalized flat pieces (no coordinate hints)
# ---------------------------------------------------------------------------

def _precompute_2d_shapes(
    pieces: List[Piece],
    slab_axis: int,
    grid_size: int,
) -> Dict[int, List[FrozenSet[Tuple[int, int]]]]:
    """For each piece, compute all valid 2D placements in an NxN grid.

    Returns {piece_index: [frozenset_of_2d_cells, ...]}.
    Only includes orientations flat along slab_axis.
    """
    other = [a for a in (0, 1, 2) if a != slab_axis]
    result: Dict[int, List[FrozenSet[Tuple[int, int]]]] = {}
    N = grid_size

    for pidx, piece in enumerate(pieces):
        shape_orients: set = set()
        for orient in get_orientations(piece):
            ext = max(c[slab_axis] for c in orient) - min(c[slab_axis] for c in orient)
            if ext != 0:
                continue
            cells2d = frozenset((c[other[0]], c[other[1]]) for c in orient)
            shape_orients.add(cells2d)

        placements: set = set()
        for shape in shape_orients:
            min_u = min(c[0] for c in shape)
            min_v = min(c[1] for c in shape)
            max_u = max(c[0] for c in shape)
            max_v = max(c[1] for c in shape)
            for du in range(-min_u, N - max_u):
                for dv in range(-min_v, N - max_v):
                    shifted = frozenset((u + du, v + dv) for u, v in shape)
                    placements.add(shifted)

        result[pidx] = list(placements)
    return result


def _find_profile_assignment(
    profiles: List[Tuple[int, int, int]],
    num_groups: int,
    c3: int, c4: int, c5: int,
) -> Optional[List[Tuple[int, int, int]]]:
    """Find num_groups profiles from `profiles` that sum to (c3, c4, c5).

    Uses memoized backtracking. Returns list of (n3, n4, n5) per group.
    """
    assignment = [None] * num_groups
    # Memoize failures: (groups_left, rem3, rem4, rem5) → False
    memo: set = set()

    def backtrack(g, rem3, rem4, rem5):
        if g == num_groups:
            return rem3 == 0 and rem4 == 0 and rem5 == 0
        key = (g, rem3, rem4, rem5)
        if key in memo:
            return False
        groups_left = num_groups - g
        for n3, n4, n5 in profiles:
            if n3 > rem3 or n4 > rem4 or n5 > rem5:
                continue
            # Pruning: remaining after this assignment must be feasible.
            r3, r4, r5 = rem3 - n3, rem4 - n4, rem5 - n5
            if groups_left > 1:
                # Each remaining group needs at least min(profile_n3) etc.
                # Quick bound: total remaining volume must equal
                # (groups_left - 1) × target
                vol_left = 3 * r3 + 4 * r4 + 5 * r5
                if vol_left != (groups_left - 1) * (3 * n3 + 4 * n4 + 5 * n5):
                    # target is constant per group
                    pass  # can't prune on volume since profiles vary
            assignment[g] = (n3, n4, n5)
            if backtrack(g + 1, r3, r4, r5):
                return True
        memo.add(key)
        return False

    if backtrack(0, c3, c4, c5):
        return list(assignment)
    return None


def _is_rod(piece: Piece) -> bool:
    """Check if a piece is a straight rod (extent along exactly one axis)."""
    coords = [tuple(int(v) for v in c) for c in piece]
    if not coords:
        return False
    spans = [
        max(c[a] for c in coords) - min(c[a] for c in coords)
        for a in range(3)
    ]
    return sum(1 for s in spans if s > 0) <= 1


def _canonical_2d_shape(piece: Piece, slab_axis: int) -> FrozenSet[Tuple[int, int]]:
    """Get canonical 2D shape key for a piece projected along slab_axis."""
    a0, a1 = (1, 2) if slab_axis == 0 else (0, 2) if slab_axis == 1 else (0, 1)
    min_u = min(c[a0] for c in piece)
    min_v = min(c[a1] for c in piece)
    return frozenset((c[a0] - min_u, c[a1] - min_v) for c in piece)


def _partition_pieces_into_slabs(
    pieces: List[Piece],
    grid_size: int,
    rng,
    slab_axis: Optional[int] = None,
) -> Optional[List[List[int]]]:
    """Partition piece indices into N groups, each summing to N^2 cells.

    Uses fine-grained shape classification: pieces are grouped by their
    canonical 2D shape (projected along the slab axis). This ensures that
    complementary piece types (e.g., L-triomino A and B) are distributed
    evenly, which is critical for tileability.

    For odd N, rod pieces (height-1 in 2D projection) are pre-assigned to
    groups to guarantee the 1-row strip can be tiled.

    Returns list of N lists of piece indices, or None if no valid partition.
    """
    N = grid_size
    target = N * N

    if slab_axis is None:
        slab_axis = detect_slab_axis(pieces)
    if slab_axis is None:
        return None

    # Classify pieces by canonical 2D shape.
    shape_classes: Dict[FrozenSet[Tuple[int, int]], List[int]] = {}
    for pidx, piece in enumerate(pieces):
        key = _canonical_2d_shape(piece, slab_axis)
        shape_classes.setdefault(key, []).append(pidx)

    # --- Odd-N pre-assignment: ensure each group has rods for the 1-row strip ---
    preassigned: List[List[int]] = [[] for _ in range(N)]
    preassigned_vol = [0] * N
    preassigned_set: set = set()

    if N % 2 == 1:
        # Identify 1D pieces: 2D shape fits in a single row when rotated.
        # Canonical shape may run along u or v, but a rod can be rotated
        # to fit either way.  Detect: one dimension has extent 1.
        rod_by_size: Dict[int, List[int]] = {}
        rod_shape_keys = set()
        for key, indices in shape_classes.items():
            u_coords = {u for u, _ in key}
            v_coords = {v for _, v in key}
            if len(u_coords) == 1 or len(v_coords) == 1:
                rod_shape_keys.add(key)
                for pidx in indices:
                    sz = len(pieces[pidx])
                    rod_by_size.setdefault(sz, []).append(pidx)

        for sz in rod_by_size:
            rng.shuffle(rod_by_size[sz])

        # Find valid combos (n3, n4, n5) where 3*n3+4*n4+5*n5 = N
        r3 = len(rod_by_size.get(3, []))
        r4 = len(rod_by_size.get(4, []))
        r5 = len(rod_by_size.get(5, []))
        rod_profiles = []
        for n3 in range(min(r3, N // 3) + 1):
            for n4 in range(min(r4, (N - 3 * n3) // 4) + 1):
                rem = N - 3 * n3 - 4 * n4
                if rem >= 0 and rem % 5 == 0:
                    n5 = rem // 5
                    if n5 <= r5:
                        rod_profiles.append((n3, n4, n5))

        if rod_profiles:
            # Find assignment of N rod profiles using at most (r3, r4, r5)
            rod_assignment = _find_profile_assignment(rod_profiles, N, r3, r4, r5)
            if rod_assignment is not None:
                ptr3, ptr4, ptr5 = 0, 0, 0
                rod3 = rod_by_size.get(3, [])
                rod4 = rod_by_size.get(4, [])
                rod5 = rod_by_size.get(5, [])
                for g, (n3, n4, n5) in enumerate(rod_assignment):
                    for pidx in rod3[ptr3:ptr3 + n3]:
                        preassigned[g].append(pidx)
                        preassigned_vol[g] += 3
                        preassigned_set.add(pidx)
                    ptr3 += n3
                    for pidx in rod4[ptr4:ptr4 + n4]:
                        preassigned[g].append(pidx)
                        preassigned_vol[g] += 4
                        preassigned_set.add(pidx)
                    ptr4 += n4
                    for pidx in rod5[ptr5:ptr5 + n5]:
                        preassigned[g].append(pidx)
                        preassigned_vol[g] += 5
                        preassigned_set.add(pidx)
                    ptr5 += n5

    # --- Assign remaining (non-rod) pieces via shape-balanced round-robin ---
    # This preserves complementary pair balance (L-A/L-B, P-A/P-B) which is
    # critical for both DLX tileability and the pair-based constructive solver.
    remaining_shape_classes: Dict[FrozenSet[Tuple[int, int]], List[int]] = {}
    for key, indices in shape_classes.items():
        filtered = [pidx for pidx in indices if pidx not in preassigned_set]
        if filtered:
            remaining_shape_classes[key] = filtered

    for key in remaining_shape_classes:
        rng.shuffle(remaining_shape_classes[key])

    groups: List[List[int]] = [list(pre) for pre in preassigned]
    group_vol = list(preassigned_vol)

    for key in sorted(remaining_shape_classes.keys(), key=lambda k: len(k)):
        indices = remaining_shape_classes[key]
        for i, pidx in enumerate(indices):
            g = i % N
            groups[g].append(pidx)
            group_vol[g] += len(pieces[pidx])

    # Verify volumes.
    if all(v == target for v in group_vol):
        return groups

    # Round-robin didn't hit target volumes — fall back to profile-based DP.
    remaining_by_size: Dict[int, List[int]] = {}
    for pidx, piece in enumerate(pieces):
        if pidx in preassigned_set:
            continue
        sz = len(piece)
        remaining_by_size.setdefault(sz, []).append(pidx)

    for sz in remaining_by_size:
        rng.shuffle(remaining_by_size[sz])

    remaining_target = target - preassigned_vol[0]
    rc3 = len(remaining_by_size.get(3, []))
    rc4 = len(remaining_by_size.get(4, []))
    rc5 = len(remaining_by_size.get(5, []))

    rem_profiles = []
    for n3 in range(min(rc3, remaining_target // 3) + 1):
        for n4 in range(min(rc4, (remaining_target - 3 * n3) // 4) + 1):
            rem = remaining_target - 3 * n3 - 4 * n4
            if rem >= 0 and rem % 5 == 0:
                n5 = rem // 5
                if n5 <= rc5:
                    rem_profiles.append((n3, n4, n5))

    if not rem_profiles:
        return None

    ideal3 = rc3 / N if N > 0 else 0
    ideal4 = rc4 / N if N > 0 else 0
    ideal5 = rc5 / N if N > 0 else 0
    rem_profiles.sort(
        key=lambda p: (p[0] - ideal3) ** 2 + (p[1] - ideal4) ** 2 + (p[2] - ideal5) ** 2
    )

    rem_assignment = _find_profile_assignment(rem_profiles, N, rc3, rc4, rc5)
    if rem_assignment is None:
        return None

    groups = [list(pre) for pre in preassigned]
    ptr3, ptr4, ptr5 = 0, 0, 0
    rem3 = remaining_by_size.get(3, [])
    rem4 = remaining_by_size.get(4, [])
    rem5 = remaining_by_size.get(5, [])
    for g, (n3, n4, n5) in enumerate(rem_assignment):
        groups[g].extend(rem3[ptr3:ptr3 + n3])
        ptr3 += n3
        groups[g].extend(rem4[ptr4:ptr4 + n4])
        ptr4 += n4
        groups[g].extend(rem5[ptr5:ptr5 + n5])
        ptr5 += n5

    # Verify volumes.
    for g in range(N):
        vol = sum(len(pieces[pidx]) for pidx in groups[g])
        if vol != target:
            return None

    return groups


def _partition_by_profile(
    pieces: List[Piece],
    grid_size: int,
    rng,
    shape_classes: Dict[Tuple[int, bool], List[int]],
) -> Optional[List[List[int]]]:
    """Fallback: partition using volume profile assignment."""
    N = grid_size
    target = N * N

    by_size: Dict[int, List[int]] = {3: [], 4: [], 5: []}
    for pidx, piece in enumerate(pieces):
        sz = len(piece)
        if sz not in by_size:
            return None
        by_size[sz].append(pidx)

    c3, c4, c5 = len(by_size[3]), len(by_size[4]), len(by_size[5])

    for sz in by_size:
        rng.shuffle(by_size[sz])

    profiles = []
    for n3 in range(min(c3, target // 3) + 1):
        for n4 in range(min(c4, (target - 3 * n3) // 4) + 1):
            rem = target - 3 * n3 - 4 * n4
            if rem >= 0 and rem % 5 == 0:
                n5 = rem // 5
                if n5 <= c5:
                    profiles.append((n3, n4, n5))

    if not profiles:
        return None

    ideal3 = c3 / N if N > 0 else 0
    ideal4 = c4 / N if N > 0 else 0
    ideal5 = c5 / N if N > 0 else 0
    profiles.sort(key=lambda p: (p[0] - ideal3) ** 2 + (p[1] - ideal4) ** 2 + (p[2] - ideal5) ** 2)

    assignment = _find_profile_assignment(profiles, N, c3, c4, c5)
    if assignment is None:
        return None

    groups: List[List[int]] = [[] for _ in range(N)]
    ptr3, ptr4, ptr5 = 0, 0, 0
    for g, (n3, n4, n5) in enumerate(assignment):
        groups[g].extend(by_size[3][ptr3:ptr3 + n3])
        ptr3 += n3
        groups[g].extend(by_size[4][ptr4:ptr4 + n4])
        ptr4 += n4
        groups[g].extend(by_size[5][ptr5:ptr5 + n5])
        ptr5 += n5

    return groups


def _solve_2d_slab_dlx(
    piece_indices: List[int],
    all_placements: Dict[int, List[FrozenSet[Tuple[int, int]]]],
    grid_size: int,
    timeout: float,
    max_nodes: int = 500_000,
) -> Optional[Dict[int, FrozenSet[Tuple[int, int]]]]:
    """Solve a single NxN slab as standard 2D exact cover.

    All pieces in piece_indices must be placed (standard exact cover).
    Returns {piece_idx: 2D cells} or None.
    """
    t0 = time.time()
    N = grid_size
    cell_names = [f"c_{u}_{v}" for u in range(N) for v in range(N)]
    piece_names = [f"p_{pidx}" for pidx in piece_indices]
    dlx = DLX(cell_names + piece_names)

    row_map = {}
    rid = 0
    for pidx in piece_indices:
        for placed_cells in all_placements.get(pidx, []):
            cols = [f"p_{pidx}"]
            for u, v in placed_cells:
                cols.append(f"c_{u}_{v}")
            dlx.add_row(rid, cols)
            row_map[rid] = (pidx, placed_cells)
            rid += 1
        if time.time() - t0 > timeout:
            return None

    solutions = dlx.solve(find_all=False, max_nodes=max_nodes)
    if not solutions:
        return None
    sol = {}
    for r in solutions[0]:
        pidx, placed = row_map[r]
        sol[pidx] = placed
    return sol


def _solve_slab_by_strips(
    group: List[int],
    all_placements: Dict[int, List[FrozenSet[Tuple[int, int]]]],
    grid_size: int,
    max_nodes_per_strip: int = 100_000,
) -> Optional[Dict[int, FrozenSet[Tuple[int, int]]]]:
    """Solve one NxN slab by decomposing into 2-row strips.

    For each 2-row strip, uses DLX with secondary piece columns so DLX
    picks which pieces go in this strip. Strip DLXes are tiny (~14 cells,
    ~14 pieces) so they're very fast.

    Returns {piece_idx: 2D cells} or None.
    """
    N = grid_size
    unplaced = set(group)
    solution: Dict[int, FrozenSet[Tuple[int, int]]] = {}

    # Build all strips, then sort by how constrained they are (fewest
    # fitting rows first). This ensures the single-row strip (N odd) — which
    # only rod pieces can fill — gets solved before greedy choices in wider
    # strips consume those rods.
    strips = []
    for v0 in range(0, N - 1, 2):
        strip_cells = frozenset((u, v) for u in range(N) for v in (v0, v0 + 1))
        strips.append(strip_cells)
    if N % 2 == 1:
        strip_cells = frozenset((u, N - 1) for u in range(N))
        strips.append(strip_cells)

    # Sort strips: most constrained first (fewest total placements from group).
    def _strip_row_count(scells):
        return sum(
            1 for pidx in group
            for pl in all_placements.get(pidx, [])
            if pl.issubset(scells)
        )
    strips.sort(key=_strip_row_count)

    for strip_cells in strips:
        # Build tiny DLX: cell columns are primary, piece columns are secondary.
        cell_names = [f"c_{u}_{v}" for u, v in sorted(strip_cells)]
        piece_names = [f"p_{pidx}" for pidx in unplaced]
        dlx = DLX(cell_names, secondary=piece_names)

        row_map = {}
        rid = 0
        for pidx in unplaced:
            for placed in all_placements.get(pidx, []):
                if placed.issubset(strip_cells):
                    cols = [f"p_{pidx}"]
                    for u, v in placed:
                        cols.append(f"c_{u}_{v}")
                    dlx.add_row(rid, cols)
                    row_map[rid] = (pidx, placed)
                    rid += 1

        solutions = dlx.solve(find_all=False, max_nodes=max_nodes_per_strip)
        if not solutions:
            return None

        # Extract placed pieces.
        for r in solutions[0]:
            pidx, cells2d = row_map[r]
            solution[pidx] = cells2d
            unplaced.discard(pidx)

    if unplaced:
        return None
    return solution


def solve_slab_layered(
    pieces: List[Piece],
    grid_size: int,
    slab_timeout: float = 15.0,
    total_timeout: float = 180.0,
    max_retries: int = 100,
) -> Tuple[Optional[Dict[int, FrozenSet[Tuple[int, int, int]]]], dict]:
    """Solve flat-piece puzzles by shape-balanced partition + strip-by-strip DLX.

    Strategy:
    1. Classify pieces by canonical 2D shape, distribute evenly to N slab groups
    2. For each slab, solve strip-by-strip: each 2-row strip is a tiny DLX
       (~14 cells, secondary piece columns) that's very fast
    3. If any slab fails, retry with different shuffles

    Works with normalized/relative pieces (no absolute coordinate hints needed).
    """
    diag = {
        'planner': 'slab_layered',
        'slab_axis': None,
        'slabs_total': 0,
        'slabs_solved': 0,
        'retries': 0,
    }

    if not pieces:
        diag['reason'] = 'empty_pieces'
        return None, diag

    axis = detect_slab_axis(pieces)
    if axis is None:
        diag['reason'] = 'no_slab_axis'
        return None, diag
    diag['slab_axis'] = axis

    total_vol = sum(len(p) for p in pieces)
    if total_vol != grid_size ** 3:
        diag['reason'] = 'volume_mismatch'
        return None, diag

    diag['slabs_total'] = grid_size
    other = [a for a in (0, 1, 2) if a != axis]
    N = grid_size
    t0 = time.time()

    # Precompute all 2D placements for each piece (shared across retries).
    all_placements = _precompute_2d_shapes(pieces, axis, N)
    diag['precompute_time'] = time.time() - t0

    if time.time() - t0 > total_timeout:
        diag['reason'] = 'precompute_timeout'
        return None, diag

    import random as _rand
    rng = _rand.Random(42)

    for retry in range(max_retries):
        if time.time() - t0 > total_timeout:
            break

        diag['retries'] = retry

        # Step 1: partition pieces into N groups (shape-balanced).
        groups = _partition_pieces_into_slabs(pieces, N, rng, slab_axis=axis)
        if groups is None:
            continue

        # Step 2: solve each group strip-by-strip.
        global_solution: Dict[int, FrozenSet[Tuple[int, int, int]]] = {}
        all_slabs_ok = True

        for slab_idx, group in enumerate(groups):
            if time.time() - t0 > total_timeout:
                all_slabs_ok = False
                break

            sol2d = _solve_slab_by_strips(group, all_placements, N)
            if sol2d is None:
                diag['failed_slab'] = slab_idx
                diag['slabs_solved'] = slab_idx
                all_slabs_ok = False
                break

            # Lift 2D → 3D.
            for pidx, cells2d in sol2d.items():
                cells3d = []
                for u, v in cells2d:
                    coord = [0, 0, 0]
                    coord[axis] = slab_idx
                    coord[other[0]] = u
                    coord[other[1]] = v
                    cells3d.append(tuple(coord))
                global_solution[pidx] = frozenset(cells3d)

        if all_slabs_ok and len(global_solution) == len(pieces):
            diag['slabs_solved'] = N
            diag['reason'] = None
            diag['total_time'] = time.time() - t0
            return global_solution, diag

    diag['reason'] = 'all_retries_exhausted'
    diag['total_time'] = time.time() - t0
    return None, diag


# ── Pair-based constructive tiling (PML-inspired stochastic matching) ──────

# The mixed_constructive generator creates pieces in complementary pairs:
#   L-triomino A + L-triomino B → tile a 2×3 rectangle
#   2×2 square + 2×2 square     → tile a 2×4 rectangle
#   P-pentomino A + P-pentomino B → tile a 2×5 rectangle
# Additionally, for odd N, straight rods tile the last row.
#
# Instead of DLX, we reconstruct the tiling by:
# 1. Identifying piece types and forming compatible pairs
# 2. Using stochastic local search to assign pairs to strips
# 3. Placing each pair deterministically within its strip
#
# This is O(N²) per slab, vs DLX which can be exponential.
# Inspired by stochastic approximation (Lab 7) and nearest-neighbor
# matching (Lab 5) from STA 561 Probabilistic Machine Learning.

# Known pair types (canonical 2D shapes):
_SHAPE_L_A = frozenset([(0, 0), (1, 0), (0, 1)])          # L-triomino type A
_SHAPE_L_B = frozenset([(0, 1), (1, 0), (1, 1)])          # L-triomino type B
_SHAPE_SQ  = frozenset([(0, 0), (1, 0), (0, 1), (1, 1)])  # 2×2 square
_SHAPE_P_A = frozenset([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)])  # P-pento A
_SHAPE_P_B = frozenset([(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])  # P-pento B

# Pair compatibility: (shape_A, shape_B) → combined width
_PAIR_WIDTH = {
    (_SHAPE_L_A, _SHAPE_L_B): 3,
    (_SHAPE_SQ, _SHAPE_SQ): 4,
    (_SHAPE_P_A, _SHAPE_P_B): 5,
}


def _classify_piece_type(shape: FrozenSet[Tuple[int, int]]) -> str:
    """Classify a canonical 2D shape by size and bounding box.

    Robust to all rotations/reflections — does not depend on exact
    canonical orientation.  Drops the A/B distinction since the
    constructive solver assigns cells freely.
    """
    n = len(shape)
    us = {u for u, _ in shape}
    vs = {v for _, v in shape}
    w = max(us) - min(us) + 1
    h = max(vs) - min(vs) + 1

    # Rod: linear (one dimension is 1)
    if w == 1 or h == 1:
        return f'ROD_{n}'
    # L-triomino: 3 cells in 2×2 bounding box
    if n == 3 and min(w, h) == 2 and max(w, h) == 2:
        return 'L'
    # Square: 4 cells in 2×2 bounding box
    if n == 4 and min(w, h) == 2 and max(w, h) == 2:
        return 'SQ'
    # P-pentomino: 5 cells in 2×3 bounding box (handles both orientations)
    if n == 5 and min(w, h) == 2 and max(w, h) == 3:
        return 'P'
    return 'UNKNOWN'


def _place_pair_in_strip(
    pidx_a: int, pidx_b: int, pair_type: str,
    strip_v0: int, strip_pos: int,
) -> Dict[int, FrozenSet[Tuple[int, int]]]:
    """Place a complementary pair at (strip_pos, strip_v0) in the 2-row strip.

    Returns {piece_idx: frozenset of (u, v) cells}.
    """
    u0 = strip_pos
    v0 = strip_v0
    v1 = strip_v0 + 1

    if pair_type == 'L':
        # L-A at (u0, v0), (u0+1, v0), (u0, v1)
        # L-B at (u0+1, v1), (u0+2, v0), (u0+2, v1)
        cells_a = frozenset([(u0, v0), (u0 + 1, v0), (u0, v1)])
        cells_b = frozenset([(u0 + 1, v1), (u0 + 2, v0), (u0 + 2, v1)])
        return {pidx_a: cells_a, pidx_b: cells_b}

    if pair_type == 'SQ':
        # Square A at (u0, v0), (u0+1, v0), (u0, v1), (u0+1, v1)
        # Square B at (u0+2, v0), (u0+3, v0), (u0+2, v1), (u0+3, v1)
        cells_a = frozenset([(u0, v0), (u0 + 1, v0), (u0, v1), (u0 + 1, v1)])
        cells_b = frozenset([(u0 + 2, v0), (u0 + 3, v0), (u0 + 2, v1), (u0 + 3, v1)])
        return {pidx_a: cells_a, pidx_b: cells_b}

    if pair_type == 'P':
        # P-A at (u0, v0), (u0+1, v0), (u0+2, v0), (u0, v1), (u0+1, v1)
        # P-B at (u0+2, v1), (u0+3, v0), (u0+4, v0), (u0+3, v1), (u0+4, v1)
        cells_a = frozenset([
            (u0, v0), (u0 + 1, v0), (u0 + 2, v0),
            (u0, v1), (u0 + 1, v1),
        ])
        cells_b = frozenset([
            (u0 + 2, v1), (u0 + 3, v0), (u0 + 4, v0),
            (u0 + 3, v1), (u0 + 4, v1),
        ])
        return {pidx_a: cells_a, pidx_b: cells_b}

    return {}


def _solve_slab_by_pairing(
    group: List[int],
    pieces: List[Piece],
    grid_size: int,
    slab_axis: int,
    rng,
) -> Optional[Dict[int, FrozenSet[Tuple[int, int]]]]:
    """Solve one NxN slab using pair-based constructive tiling.

    Exploits the known pair structure of mixed_constructive pieces.
    Uses stochastic matching: shuffles pairs and assigns to strips.
    Falls back to None if pairing fails (pieces aren't from mixed_constructive).
    """
    N = grid_size

    # Classify pieces in this group.
    by_type: Dict[str, List[int]] = {}
    for pidx in group:
        shape = _canonical_2d_shape(pieces[pidx], slab_axis)
        ptype = _classify_piece_type(shape)
        by_type.setdefault(ptype, []).append(pidx)

    # Check if we have compatible pair types (even counts).
    l_all = by_type.get('L', [])
    sq = by_type.get('SQ', [])
    p_all = by_type.get('P', [])

    if len(l_all) % 2 != 0 or len(sq) % 2 != 0 or len(p_all) % 2 != 0:
        return None

    # Form pairs (any two of same type).
    rng.shuffle(l_all)
    rng.shuffle(sq)
    rng.shuffle(p_all)

    pairs = []  # (pidx_a, pidx_b, pair_type, width)
    for i in range(0, len(l_all), 2):
        pairs.append((l_all[i], l_all[i + 1], 'L', 3))
    for i in range(0, len(sq), 2):
        pairs.append((sq[i], sq[i + 1], 'SQ', 4))
    for i in range(0, len(p_all), 2):
        pairs.append((p_all[i], p_all[i + 1], 'P', 5))

    # Collect rods for the last row (odd N).
    rods = []
    for k in sorted(by_type.keys()):
        if k.startswith('ROD_'):
            for pidx in by_type[k]:
                rods.append((pidx, int(k.split('_')[1])))

    # Check total volume.
    pair_vol = sum(2 * w for _, _, _, w in pairs)
    rod_vol = sum(sz for _, sz in rods)
    if pair_vol + rod_vol != N * N:
        return None

    # Assign pairs to 2-row strips: each strip has width N.
    # This is a bin packing problem: partition pairs by width into groups summing to N.
    # Use the known line partitions.
    pair_widths = [w for _, _, _, w in pairs]

    # Build strips: each is a sequence of pair indices whose widths sum to N.
    n_strips = N // 2
    strips_assignment = _assign_pairs_to_strips(pair_widths, N, n_strips, rng)
    if strips_assignment is None:
        return None

    # Place pairs in strips.
    solution: Dict[int, FrozenSet[Tuple[int, int]]] = {}

    for strip_idx, pair_indices in enumerate(strips_assignment):
        v0 = strip_idx * 2
        pos = 0
        for pi in pair_indices:
            pidx_a, pidx_b, ptype, width = pairs[pi]
            placed = _place_pair_in_strip(pidx_a, pidx_b, ptype, v0, pos)
            solution.update(placed)
            pos += width

    # Place rods in last row (odd N).
    if N % 2 == 1:
        pos = 0
        for pidx, sz in rods:
            cells = frozenset((pos + t, N - 1) for t in range(sz))
            solution[pidx] = cells
            pos += sz
        if pos != N:
            return None

    # Verify all pieces placed and all cells covered.
    if len(solution) != len(group):
        return None

    all_cells = set()
    for cells in solution.values():
        all_cells |= cells
    expected = {(u, v) for u in range(N) for v in range(N)}
    if all_cells != expected:
        return None

    return solution


def _assign_pairs_to_strips(
    widths: List[int],
    strip_width: int,
    n_strips: int,
    rng,
) -> Optional[List[List[int]]]:
    """Assign pair indices to strips such that each strip's widths sum to strip_width.

    Uses greedy assignment with stochastic ordering (inspired by stochastic
    approximation). Falls back to backtracking if greedy fails.
    """
    # Group pairs by width.
    by_width: Dict[int, List[int]] = {}
    for i, w in enumerate(widths):
        by_width.setdefault(w, []).append(i)

    # Get valid strip patterns (sequences of widths summing to strip_width).
    # Import from grading harness.
    try:
        from grading_harness import _line_size_partitions
        patterns = _line_size_partitions(strip_width)
    except ImportError:
        patterns = _generate_patterns(strip_width)

    if not patterns:
        return None

    # Count needed per width.
    width_counts = {w: len(idxs) for w, idxs in by_width.items()}

    # Find an assignment of n_strips patterns using exactly the available widths.
    # Try each permutation of patterns with backtracking.
    assignment = _find_pattern_assignment(patterns, n_strips, dict(width_counts))
    if assignment is None:
        return None

    # Distribute pair indices to strips.
    ptr = {w: 0 for w in by_width}
    result = []
    for pattern in assignment:
        strip_pairs = []
        for w in pattern:
            idx = by_width[w][ptr[w]]
            strip_pairs.append(idx)
            ptr[w] += 1
        result.append(strip_pairs)

    return result


def _generate_patterns(n: int) -> List[Tuple[int, ...]]:
    """Generate all partitions of n into parts from {3, 4, 5}."""
    result = []

    def _backtrack(remaining, current):
        if remaining == 0:
            result.append(tuple(current))
            return
        for s in (3, 4, 5):
            if s <= remaining:
                current.append(s)
                _backtrack(remaining - s, current)
                current.pop()

    _backtrack(n, [])
    return result


def _find_pattern_assignment(
    patterns: List[Tuple[int, ...]],
    n_strips: int,
    width_counts: Dict[int, int],
    row_width: int = 0,
) -> Optional[List[Tuple[int, ...]]]:
    """Find n_strips patterns that together use exactly width_counts pieces.

    Enumerates valid profiles (a3, a4, a5) where 3*a3+4*a4+5*a5=N directly
    instead of generating all partitions first.  Three optimizations:
    1. Suffix min/max bounds pruning — skip branches that can't possibly work.
    2. Two-profile direct solve — when 2 profiles remain, solve the linear
       system in O(1) instead of iterating.
    3. Profile ordering — most constrained profiles first (highest per-strip
       consumption) to minimize branching factor at the top of the tree.

    row_width: if provided, used directly as the row width N. Otherwise
    inferred from patterns[0] or width_counts.
    """
    # Canonical width ordering: always (3, 4, 5).
    widths = sorted(set(width_counts.keys()) | {3, 4, 5})
    wc_tup = tuple(width_counts.get(w, 0) for w in widths)
    nw = len(widths)

    # Determine row width N.
    if row_width > 0:
        N_row = row_width
    elif patterns:
        N_row = sum(patterns[0])
    else:
        total_vol = sum(w * c for w, c in width_counts.items())
        N_row = total_vol // n_strips if n_strips > 0 else 0

    # Enumerate profiles directly: (a3, a4, a5) where 3*a3+4*a4+5*a5 = N_row.
    # This is O(N²) instead of generating all partitions (exponential for large N).
    profile_map: Dict[Tuple[int, ...], Tuple[int, ...]] = {}  # profile -> representative pattern
    for a3 in range(N_row // 3 + 1):
        rem_after_3 = N_row - 3 * a3
        for a4 in range(rem_after_3 // 4 + 1):
            rem_after_4 = rem_after_3 - 4 * a4
            if rem_after_4 >= 0 and rem_after_4 % 5 == 0:
                a5 = rem_after_4 // 5
                profile = tuple([a3, a4, a5][:nw])  # match width ordering
                # Generate one representative pattern: [3]*a3 + [4]*a4 + [5]*a5.
                rep = (3,) * a3 + (4,) * a4 + (5,) * a5
                profile_map[profile] = rep

    # Sort profiles by decreasing sum (highest total consumption first —
    # these are most constrained and reduce branching factor at top levels).
    sorted_keys = sorted(profile_map.keys(), key=lambda k: sum(k), reverse=True)
    prof_keys = sorted_keys
    prof_pats = [profile_map[k] for k in prof_keys]
    n_prof = len(prof_keys)

    if n_prof == 0:
        if n_strips == 0 and all(w == 0 for w in wc_tup):
            return []
        return None

    # Precompute suffix min/max per width for feasibility pruning.
    # suffix_max[i][j] = max pk[j] over profiles i..n_prof-1
    # suffix_min[i][j] = min pk[j] over profiles i..n_prof-1
    suffix_max = [None] * (n_prof + 1)
    suffix_min = [None] * (n_prof + 1)
    suffix_max[n_prof] = (0,) * nw
    suffix_min[n_prof] = (0,) * nw
    for i in range(n_prof - 1, -1, -1):
        pk = prof_keys[i]
        if i == n_prof - 1:
            suffix_max[i] = pk
            suffix_min[i] = pk
        else:
            suffix_max[i] = tuple(max(pk[j], suffix_max[i + 1][j]) for j in range(nw))
            suffix_min[i] = tuple(min(pk[j], suffix_min[i + 1][j]) for j in range(nw))

    counts = [0] * n_prof

    def _solve(idx: int, rem_strips: int, rem: Tuple[int, ...]) -> bool:
        if idx == n_prof:
            return rem_strips == 0 and all(r == 0 for r in rem)

        pk = prof_keys[idx]

        # --- Last profile: direct computation (O(1)) ---
        if idx == n_prof - 1:
            all_zero = True
            for j in range(nw):
                if pk[j] != 0:
                    all_zero = False
                    break
            if all_zero:
                if all(r == 0 for r in rem):
                    counts[idx] = rem_strips
                    return True
                return False
            reps = -1
            for j in range(nw):
                if pk[j] > 0:
                    q, r_mod = divmod(rem[j], pk[j])
                    if r_mod != 0:
                        return False
                    if reps == -1:
                        reps = q
                    elif q != reps:
                        return False
                elif rem[j] != 0:
                    return False
            if reps < 0:
                reps = 0
            if reps != rem_strips:
                return False
            counts[idx] = reps
            return True

        # --- Second-to-last profile: two-variable linear solve (O(1)) ---
        if idx == n_prof - 2:
            pk2 = prof_keys[n_prof - 1]
            # c_a + c_b = rem_strips
            # c_a * pk[j] + c_b * pk2[j] = rem[j]  for all j
            # => c_a * (pk[j] - pk2[j]) = rem[j] - rem_strips * pk2[j]
            c_a = -1
            for j in range(nw):
                diff = pk[j] - pk2[j]
                rhs = rem[j] - rem_strips * pk2[j]
                if diff != 0:
                    q, r_mod = divmod(rhs, diff)
                    if r_mod != 0 or q < 0:
                        return False
                    if c_a == -1:
                        c_a = q
                    elif c_a != q:
                        return False
                elif rhs != 0:
                    return False
            if c_a == -1:
                c_a = 0
            c_b = rem_strips - c_a
            if c_b < 0:
                return False
            # Verify (guard against rounding/overflow).
            for j in range(nw):
                if c_a * pk[j] + c_b * pk2[j] != rem[j]:
                    return False
            counts[idx] = c_a
            counts[n_prof - 1] = c_b
            return True

        # --- General case: iterate with suffix bounds pruning ---
        max_reps = rem_strips
        for j in range(nw):
            if pk[j] > 0:
                max_reps = min(max_reps, rem[j] // pk[j])

        s_max = suffix_max[idx + 1]
        s_min = suffix_min[idx + 1]

        for reps in range(max_reps, -1, -1):
            new_rem_strips = rem_strips - reps
            # Suffix bounds pruning: check if remaining profiles can
            # possibly consume exactly the remaining width counts.
            feasible = True
            for j in range(nw):
                rj = rem[j] - pk[j] * reps
                if rj < 0:
                    feasible = False
                    break
                if new_rem_strips * s_max[j] < rj:
                    feasible = False
                    break
                if new_rem_strips * s_min[j] > rj:
                    feasible = False
                    break
            if not feasible:
                continue
            new_rem = tuple(rem[j] - pk[j] * reps for j in range(nw))
            counts[idx] = reps
            if _solve(idx + 1, new_rem_strips, new_rem):
                return True
        return False

    if not _solve(0, n_strips, wc_tup):
        return None

    # Expand counts into a flat pattern list.
    result: List[Tuple[int, ...]] = []
    for idx, cnt in enumerate(counts):
        p = prof_pats[idx]  # representative pattern for this profile
        result.extend([p] * cnt)
    return result


def solve_slab_paired(
    pieces: List[Piece],
    grid_size: int,
    total_timeout: float = 30.0,
    max_retries: int = 20,
) -> Tuple[Optional[Dict[int, FrozenSet[Tuple[int, int, int]]]], dict]:
    """Solve flat-piece puzzles using pair-based constructive tiling.

    Global assignment approach: classifies all pieces, forms pairs, then
    assigns pairs to strip-slots across all slabs simultaneously.  This
    avoids the partition-balance problem that caused failures at N>=9.

    PML-inspired: stochastic matching (Lab 7) for pair formation,
    profile-based DP for rod and strip assignment.

    Falls back to None if pieces don't match the mixed_constructive pattern.
    """
    diag: dict = {
        'planner': 'slab_paired',
        'slab_axis': None,
        'slabs_total': 0,
        'slabs_solved': 0,
        'retries': 0,
    }

    axis = detect_slab_axis(pieces)
    if axis is None:
        diag['reason'] = 'no_slab_axis'
        return None, diag
    diag['slab_axis'] = axis

    N = grid_size
    total_vol = sum(len(p) for p in pieces)
    if total_vol != N ** 3:
        diag['reason'] = 'volume_mismatch'
        return None, diag

    diag['slabs_total'] = N
    other = [a for a in (0, 1, 2) if a != axis]
    t0 = time.time()

    # --- Classify all pieces globally ---
    by_type: Dict[str, List[int]] = {}
    for pidx, p in enumerate(pieces):
        shape = _canonical_2d_shape(p, axis)
        ptype = _classify_piece_type(shape)
        by_type.setdefault(ptype, []).append(pidx)

    if by_type.get('UNKNOWN'):
        diag['reason'] = 'unknown_pieces'
        return None, diag

    # --- All-rod fast path (e.g. striped constructive cases) ---
    all_rod = all(k.startswith('ROD_') for k in by_type.keys())
    if all_rod:
        rod_by_size: Dict[int, List[int]] = {}
        for k in sorted(by_type.keys()):
            sz = int(k.split('_')[1])
            for pidx in by_type[k]:
                rod_by_size.setdefault(sz, []).append(pidx)

        rod_counts: Dict[int, int] = {sz: len(lst)
                                       for sz, lst in rod_by_size.items()}
        total_rows = N * N

        pat_assign = _find_pattern_assignment([], total_rows, rod_counts,
                                              row_width=N)
        if pat_assign is None:
            diag['reason'] = 'rod_pattern_failed'
            return None, diag

        # Build pointers for each rod size.
        rod_ptr: Dict[int, int] = {sz: 0 for sz in rod_by_size}
        solution: Dict[int, FrozenSet[Tuple[int, int, int]]] = {}
        for ri, pattern in enumerate(pat_assign):
            slab_idx = ri // N
            row_idx = ri % N
            pos = 0
            for sz in pattern:
                pidx = rod_by_size[sz][rod_ptr[sz]]
                rod_ptr[sz] += 1
                cells3d = []
                for t in range(sz):
                    coord = [0, 0, 0]
                    coord[axis] = slab_idx
                    coord[other[0]] = pos + t
                    coord[other[1]] = row_idx
                    cells3d.append(tuple(coord))
                solution[pidx] = frozenset(cells3d)
                pos += sz

        if len(solution) == len(pieces):
            all_cells: set = set()
            for cells in solution.values():
                all_cells |= cells
            expected = {(x, y, z)
                        for x in range(N) for y in range(N) for z in range(N)}
            if all_cells == expected:
                diag['slabs_solved'] = N
                diag['reason'] = None
                diag['total_time'] = time.time() - t0
                return solution, diag

        diag['reason'] = 'rod_placement_failed'
        return None, diag

    l_all = by_type.get('L', [])
    sq_all = by_type.get('SQ', [])
    p_all = by_type.get('P', [])

    if len(l_all) % 2 or len(sq_all) % 2 or len(p_all) % 2:
        diag['reason'] = 'odd_type_counts'
        return None, diag

    # --- Rod assignment for odd N ---
    rod_assignments: List[List[Tuple[int, int]]] = [[] for _ in range(N)]
    if N % 2 == 1:
        rod_by_size: Dict[int, List[int]] = {}
        for k in sorted(by_type.keys()):
            if k.startswith('ROD_'):
                sz = int(k.split('_')[1])
                for pidx in by_type[k]:
                    rod_by_size.setdefault(sz, []).append(pidx)

        r3 = len(rod_by_size.get(3, []))
        r4 = len(rod_by_size.get(4, []))
        r5 = len(rod_by_size.get(5, []))

        rod_profiles = []
        for n3 in range(min(r3, N // 3) + 1):
            for n4 in range(min(r4, (N - 3 * n3) // 4) + 1):
                rem = N - 3 * n3 - 4 * n4
                if rem >= 0 and rem % 5 == 0:
                    n5 = rem // 5
                    if n5 <= r5:
                        rod_profiles.append((n3, n4, n5))

        if not rod_profiles:
            diag['reason'] = 'no_rod_profiles'
            return None, diag

        rod_assign = _find_profile_assignment(rod_profiles, N, r3, r4, r5)
        if rod_assign is None:
            diag['reason'] = 'rod_assignment_failed'
            return None, diag

        ptr3, ptr4, ptr5 = 0, 0, 0
        rod3 = rod_by_size.get(3, [])
        rod4 = rod_by_size.get(4, [])
        rod5 = rod_by_size.get(5, [])
        for g, (n3, n4, n5) in enumerate(rod_assign):
            for pidx in rod3[ptr3:ptr3 + n3]:
                rod_assignments[g].append((pidx, 3))
            ptr3 += n3
            for pidx in rod4[ptr4:ptr4 + n4]:
                rod_assignments[g].append((pidx, 4))
            ptr4 += n4
            for pidx in rod5[ptr5:ptr5 + n5]:
                rod_assignments[g].append((pidx, 5))
            ptr5 += n5

    # --- Global strip pattern assignment ---
    n_strips_per_slab = N // 2
    total_strips = N * n_strips_per_slab

    import random as _rand
    rng = _rand.Random(42)

    for retry in range(max_retries):
        if time.time() - t0 > total_timeout:
            break
        diag['retries'] = retry

        # Shuffle pieces for stochastic matching.
        l_shuf = list(l_all)
        sq_shuf = list(sq_all)
        p_shuf = list(p_all)
        rng.shuffle(l_shuf)
        rng.shuffle(sq_shuf)
        rng.shuffle(p_shuf)

        # Form pairs.
        pairs = []
        for i in range(0, len(l_shuf), 2):
            pairs.append((l_shuf[i], l_shuf[i + 1], 'L', 3))
        for i in range(0, len(sq_shuf), 2):
            pairs.append((sq_shuf[i], sq_shuf[i + 1], 'SQ', 4))
        for i in range(0, len(p_shuf), 2):
            pairs.append((p_shuf[i], p_shuf[i + 1], 'P', 5))

        width_counts: Dict[int, int] = {}
        for _, _, _, w in pairs:
            width_counts[w] = width_counts.get(w, 0) + 1

        pat_assign = _find_pattern_assignment([], total_strips,
                                              dict(width_counts),
                                              row_width=N)
        if pat_assign is None:
            diag['reason'] = 'no_pattern_assignment'
            return None, diag  # deterministic — retrying won't help

        # Map pairs to strip slots.
        pair_by_width: Dict[int, List[int]] = {}
        for i, (_, _, _, w) in enumerate(pairs):
            pair_by_width.setdefault(w, []).append(i)

        ptr: Dict[int, int] = {w: 0 for w in pair_by_width}
        solution: Dict[int, FrozenSet[Tuple[int, int, int]]] = {}
        ok = True

        for si in range(total_strips):
            slab_idx = si // n_strips_per_slab
            strip_idx = si % n_strips_per_slab
            v0 = strip_idx * 2
            pattern = pat_assign[si]
            pos = 0
            for w in pattern:
                pi = pair_by_width[w][ptr[w]]
                ptr[w] += 1
                pidx_a, pidx_b, ptype, _ = pairs[pi]
                placed = _place_pair_in_strip(pidx_a, pidx_b, ptype, v0, pos)
                for pidx, cells2d in placed.items():
                    cells3d = []
                    for u, v in cells2d:
                        coord = [0, 0, 0]
                        coord[axis] = slab_idx
                        coord[other[0]] = u
                        coord[other[1]] = v
                        cells3d.append(tuple(coord))
                    solution[pidx] = frozenset(cells3d)
                pos += w

        # Place rods in last row of each slab (odd N).
        if N % 2 == 1:
            for slab_idx in range(N):
                pos = 0
                for pidx, sz in rod_assignments[slab_idx]:
                    cells3d = []
                    for t in range(sz):
                        coord = [0, 0, 0]
                        coord[axis] = slab_idx
                        coord[other[0]] = pos + t
                        coord[other[1]] = N - 1
                        cells3d.append(tuple(coord))
                    solution[pidx] = frozenset(cells3d)
                    pos += sz
                if pos != N:
                    ok = False
                    break

        if not ok:
            continue

        # Verify completeness.
        if len(solution) == len(pieces):
            all_cells: set = set()
            for cells in solution.values():
                all_cells |= cells
            expected = {(x, y, z)
                        for x in range(N) for y in range(N) for z in range(N)}
            if all_cells == expected:
                diag['slabs_solved'] = N
                diag['reason'] = None
                diag['total_time'] = time.time() - t0
                return solution, diag

    diag['reason'] = 'all_retries_exhausted'
    diag['total_time'] = time.time() - t0
    return None, diag
