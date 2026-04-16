"""
Flexible block decomposition planner for large polycube packing (v3).

Decomposes a 12^3 (or any N^3 where block_size divides N) grid into
block_size^3 sub-cubes and solves them in sweep order.  Pieces are
allowed to "leak" across block boundaries into blocks that have not
yet been processed.

Algorithm
---------
1. Pre-compute all orientations for every piece (cached).
2. Sweep blocks in a fixed order.  For each block:
   a. Identify empty cells within the block.
   b. Enumerate candidate placements from ALL unplaced pieces, but
      cap at max_placements_per_piece to keep DLX small.
      Placements are ranked by overlap with the block (cells inside /
      total cells) — prefer pieces that mostly sit in this block.
   c. Build a DLX with primary + secondary columns.
   d. Solve.  Apply globally.
3. Multiple trials with shuffled block orders.

The per-piece placement cap is the key scalability lever.  With ~460
pieces and cap=10, the DLX has at most ~4600 rows on 64 primary
columns — tiny and fast.

Piece size: 3, 4, or 5 (professor spec).
Block size: 4 recommended for 12^3 (27 blocks, 64 cells each).
"""

from __future__ import annotations

import random
import time
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from phase1.polycube import get_orientations


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DLX with primary / secondary column support
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _Node:
    __slots__ = ('left', 'right', 'up', 'down', 'column', 'row_id')
    def __init__(self, column=None, row_id=None):
        self.left = self; self.right = self
        self.up = self; self.down = self
        self.column = column; self.row_id = row_id

class _Column(_Node):
    __slots__ = ('size', 'name')
    def __init__(self, name=None):
        super().__init__()
        self.column = self; self.size = 0; self.name = name

class _DLX:
    """DLX with primary + secondary columns."""
    def __init__(self, primary_names, secondary_names=()):
        self.header = _Column("header")
        self.columns: Dict[str, _Column] = {}
        self.solution = []; self.solutions = []
        prev = self.header
        for name in primary_names:
            col = _Column(name); self.columns[name] = col
            col.right = self.header; col.left = prev
            prev.right = col; self.header.left = col; prev = col
        for name in secondary_names:
            col = _Column(name); col.left = col; col.right = col
            self.columns[name] = col

    def add_row(self, row_id, col_names):
        first = prev = None
        for cname in col_names:
            col = self.columns[cname]
            node = _Node(column=col, row_id=row_id)
            node.down = col; node.up = col.up
            col.up.down = node; col.up = node; col.size += 1
            if first is None: first = prev = node
            else:
                node.left = prev; node.right = first
                prev.right = node; first.left = node; prev = node

    def _cover(self, col):
        col.right.left = col.left; col.left.right = col.right
        row = col.down
        while row is not col:
            j = row.right
            while j is not row:
                j.down.up = j.up; j.up.down = j.down; j.column.size -= 1
                j = j.right
            row = row.down

    def _uncover(self, col):
        row = col.up
        while row is not col:
            j = row.left
            while j is not row:
                j.column.size += 1; j.down.up = j; j.up.down = j; j = j.left
            row = row.up
        col.right.left = col; col.left.right = col

    def _choose_column(self):
        best = None; min_size = float('inf')
        col = self.header.right
        while col is not self.header:
            if col.size < min_size:
                min_size = col.size; best = col
                if min_size == 0: break
            col = col.right
        return best

    def solve(self, find_all=False):
        self.solutions = []; self.solution = []; self._search(find_all)
        return self.solutions

    def _search(self, find_all):
        if self.header.right is self.header:
            self.solutions.append(list(self.solution)); return True
        col = self._choose_column()
        if col.size == 0: return False
        self._cover(col)
        row = col.down
        while row is not col:
            self.solution.append(row.row_id)
            j = row.right
            while j is not row: self._cover(j.column); j = j.right
            found = self._search(find_all)
            if found and not find_all:
                j = row.left
                while j is not row: self._uncover(j.column); j = j.left
                self._uncover(col); return True
            self.solution.pop()
            j = row.left
            while j is not row: self._uncover(j.column); j = j.left
            row = row.down
        self._uncover(col); return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cache_orientations(pieces):
    """Pre-compute orientations + bounding boxes."""
    cache = []
    for piece in pieces:
        orients = get_orientations(piece)
        entries = []
        for orient in orients:
            cells = tuple(sorted(orient))
            mx = max(c[0] for c in cells)
            my = max(c[1] for c in cells)
            mz = max(c[2] for c in cells)
            entries.append((cells, mx, my, mz))
        cache.append(entries)
    return cache


def _block_cells(bx, by, bz, bs):
    return frozenset(
        (x, y, z)
        for x in range(bx * bs, (bx + 1) * bs)
        for y in range(by * bs, (by + 1) * bs)
        for z in range(bz * bs, (bz + 1) * bs)
    )


def _gen_placements_ranked(
    orient_entry,
    grid_size, block_min, block_max,
    occupied, empty_in_block,
    max_placements,
):
    """Generate placements of one piece touching this block.

    Returns at most max_placements, ranked by overlap ratio (cells in
    block / total cells, descending).  This prioritizes placements
    where the piece sits mostly inside the block.
    """
    bx0, by0, bz0 = block_min
    bx1, by1, bz1 = block_max
    candidates = []  # (overlap_count, frozenset)

    for cells, mx, my, mz in orient_entry:
        psize = len(cells)
        dx_lo = max(0, bx0 - mx)
        dx_hi = min(grid_size - 1 - mx, bx1 - 1)
        dy_lo = max(0, by0 - my)
        dy_hi = min(grid_size - 1 - my, by1 - 1)
        dz_lo = max(0, bz0 - mz)
        dz_hi = min(grid_size - 1 - mz, bz1 - 1)
        if dx_lo > dx_hi or dy_lo > dy_hi or dz_lo > dz_hi:
            continue

        for dx in range(dx_lo, dx_hi + 1):
            for dy in range(dy_lo, dy_hi + 1):
                for dz in range(dz_lo, dz_hi + 1):
                    placed = frozenset(
                        (x + dx, y + dy, z + dz) for x, y, z in cells
                    )
                    in_block = placed & empty_in_block
                    if not in_block:
                        continue
                    if placed & occupied:
                        continue
                    candidates.append((len(in_block), placed))

    # Sort by overlap descending, keep top-K
    candidates.sort(key=lambda x: -x[0])
    return [pl for _, pl in candidates[:max_placements]]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-block DLX solve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _solve_block_dlx(empty_cells, candidate_rows):
    """DLX solve for one block.

    candidate_rows: list of (pidx, frozenset)
    Returns list of (pidx, cells) or None.
    """
    if not empty_cells:
        return []

    primary = sorted(f"c_{x}_{y}_{z}" for x, y, z in empty_cells)
    primary_set = set(empty_cells)

    piece_ids = sorted({pidx for pidx, _ in candidate_rows})
    outside_cells: Set[Tuple[int, int, int]] = set()
    for _, placed in candidate_rows:
        outside_cells.update(placed - primary_set)

    sec_piece = [f"p_{pidx}" for pidx in piece_ids]
    sec_cell = sorted(f"c_{x}_{y}_{z}" for x, y, z in outside_cells)
    secondary = sec_piece + sec_cell

    dlx = _DLX(primary, secondary)
    row_map = {}
    for rid, (pidx, placed) in enumerate(candidate_rows):
        col_names = [f"p_{pidx}"]
        for x, y, z in placed:
            col_names.append(f"c_{x}_{y}_{z}")
        dlx.add_row(rid, col_names)
        row_map[rid] = (pidx, placed)

    solutions = dlx.solve(find_all=False)
    if not solutions:
        return None
    return [row_map[rid] for rid in solutions[0]]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Block sweep
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _try_sweep(
    pieces, grid_size, block_size,
    block_order,
    orient_cache,
    max_plmts_per_piece,
    timeout, t0,
    verbose=False,
):
    """One sweep attempt. Returns solution dict or None."""
    occupied: Set[Tuple[int, int, int]] = set()
    placed: Set[int] = set()
    global_solution: Dict[int, FrozenSet] = {}

    for block_idx, (bx, by, bz) in enumerate(block_order):
        if time.time() - t0 > timeout:
            return None

        b_cells = _block_cells(bx, by, bz, block_size)
        empty = b_cells - occupied
        if not empty:
            continue

        bmin = (bx * block_size, by * block_size, bz * block_size)
        bmax = ((bx+1)*block_size, (by+1)*block_size, (bz+1)*block_size)

        candidate_rows: List[Tuple[int, FrozenSet]] = []
        for pidx in range(len(pieces)):
            if pidx in placed:
                continue
            plmts = _gen_placements_ranked(
                orient_cache[pidx], grid_size,
                bmin, bmax, occupied, empty,
                max_plmts_per_piece,
            )
            for pl in plmts:
                candidate_rows.append((pidx, pl))

        if not candidate_rows:
            if verbose:
                print(f"    block ({bx},{by},{bz}): "
                      f"{len(empty)} empty, 0 cands — fail")
            return None

        if verbose and len(candidate_rows) > 5000:
            print(f"    block ({bx},{by},{bz}): "
                  f"{len(empty)} empty, {len(candidate_rows)} cands...")

        solution = _solve_block_dlx(empty, candidate_rows)
        if solution is None:
            if verbose:
                print(f"    block ({bx},{by},{bz}): "
                      f"{len(empty)} empty, {len(candidate_rows)} cands, "
                      f"DLX failed")
            return None

        for pidx, cells in solution:
            placed.add(pidx)
            global_solution[pidx] = cells
            occupied.update(cells)

    if len(placed) != len(pieces):
        return None
    return global_solution


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def solve_flexible_blocks(
    pieces,
    grid_size: int,
    block_size: int = 4,
    max_trials: int = 15,
    max_placements_per_piece: int = 8,
    timeout: float = 120.0,
    verbose: bool = False,
) -> Tuple[Optional[Dict[int, FrozenSet]], dict]:
    """Flexible block decomposition with forward-leak.

    Args:
        pieces: list of pieces, each a list/set of (x,y,z) cells.
        grid_size: side length of the target cube.
        block_size: sub-cube side. Must divide grid_size.
        max_trials: random-restart attempts.
        max_placements_per_piece: cap on ranked placements per piece per
            block.  Controls DLX size vs coverage tradeoff.
        timeout: wall-clock cap in seconds.
        verbose: print progress.

    Returns:
        (solution_dict, diagnostics)
    """
    t0 = time.time()

    total_vol = sum(len(p) for p in pieces)
    if total_vol != grid_size ** 3:
        return None, {'reason': 'volume_mismatch'}

    if grid_size % block_size != 0:
        for alt in (3, 4, 6):
            if grid_size % alt == 0:
                block_size = alt
                break
        else:
            return None, {'reason': 'no_divisible_block_size'}

    nblk = grid_size // block_size
    orient_cache = _cache_orientations(pieces)

    base_order = [
        (bx, by, bz)
        for bz in range(nblk) for by in range(nblk) for bx in range(nblk)
    ]

    # Also try axis-sweep orders (x-first, y-first)
    alt_orders = [
        base_order,
        [(bx,by,bz) for bx in range(nblk) for by in range(nblk) for bz in range(nblk)],
        [(bx,by,bz) for by in range(nblk) for bz in range(nblk) for bx in range(nblk)],
    ]

    rng = random.Random(grid_size * 7919 + len(pieces) * 4217)

    best_diag = {
        'reason': 'all_trials_failed', 'grid_size': grid_size,
        'block_size': block_size, 'nblk': nblk, 'trials_attempted': 0,
    }

    for trial in range(max_trials):
        if time.time() - t0 > timeout:
            best_diag['reason'] = 'timeout'
            best_diag['trials_attempted'] = trial
            break

        # Use deterministic axis-sweeps first, then random shuffles
        if trial < len(alt_orders):
            order = list(alt_orders[trial])
        else:
            order = list(base_order)
            rng.shuffle(order)

        # Vary placement cap across trials for diversity
        cap = max_placements_per_piece
        if trial >= 3:
            cap = rng.choice([5, 8, 12, 16, 20])

        if verbose:
            print(f"  v3 trial {trial}: cap={cap}, "
                  f"order={'axis'+str(trial) if trial < 3 else 'shuffled'}")

        result = _try_sweep(
            pieces, grid_size, block_size, order,
            orient_cache, cap, timeout, t0, verbose,
        )

        if result is not None:
            return result, {
                'reason': None, 'trial': trial,
                'grid_size': grid_size, 'block_size': block_size,
                'max_plmts_per_piece': cap,
                'time': time.time() - t0,
            }

        best_diag['trials_attempted'] = trial + 1

    best_diag['time'] = time.time() - t0
    return None, best_diag


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Solution verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def verify_solution(solution, pieces, grid_size):
    """Strictly verify a block planner solution."""
    if solution is None:
        return False, "no solution"
    if len(solution) != len(pieces):
        return False, f"placed {len(solution)}/{len(pieces)} pieces"

    from phase1.polycube import normalize
    all_cells: Set[Tuple[int, int, int]] = set()
    for pidx, cells in solution.items():
        cells_set = set(cells)
        for x, y, z in cells_set:
            if not (0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size):
                return False, f"piece {pidx} out of bounds"
        placed_norm = normalize(cells_set)
        valid_orients = get_orientations(pieces[pidx])
        if placed_norm not in valid_orients:
            return False, f"piece {pidx} shape mismatch"
        overlap = all_cells & cells_set
        if overlap:
            return False, f"piece {pidx} overlaps"
        all_cells.update(cells_set)
    if len(all_cells) != grid_size ** 3:
        return False, f"covered {len(all_cells)}/{grid_size**3}"
    return True, "valid"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Smoke test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import sys, io, contextlib
    sys.path.insert(0, ".")
    from phase2.data_generator import enumerate_polycubes, generate_puzzle_instances
    from fixture_a_plus import load_fixture

    print("=== Block Planner V3 Smoke Test ===\n")

    print("--- 4x4x4 ---")
    catalog = enumerate_polycubes(max_size=5)
    with contextlib.redirect_stdout(io.StringIO()):
        insts = generate_puzzle_instances(
            num_instances=1, grid_size=4, polycube_catalog=catalog,
            min_piece_size=3, max_piece_size=5, dlx_timeout=10.0, seed=42,
        )
    pieces_4 = insts[0]["pieces"]
    sol, diag = solve_flexible_blocks(
        pieces_4, 4, block_size=4, verbose=True, timeout=15,
    )
    if sol:
        ok, msg = verify_solution(sol, pieces_4, 4)
        print(f"  {msg} ({diag.get('time',0):.2f}s)\n")
    else:
        print(f"  Failed: {diag}\n")

    print("--- 12x12x12 fixture cases (first 5) ---")
    cases = load_fixture()
    cases_12 = [c for c in cases if c.grid_size == 12 and c.expected_solvable][:5]
    for c in cases_12:
        print(f"\n  {c.case_id} ({c.generator}, {len(c.pieces)} pieces)")
        sol, diag = solve_flexible_blocks(
            c.pieces, 12, block_size=4, max_trials=10,
            max_placements_per_piece=10, timeout=60,
            verbose=True,
        )
        if sol:
            ok, msg = verify_solution(sol, c.pieces, 12)
            print(f"    {msg} ({diag.get('time',0):.2f}s, trial {diag.get('trial')})")
        else:
            print(f"    Failed: {diag}")
