"""
Hybrid solver: NN-guided beam search first, DLX exact cover fallback.

Strategy:
1. Quick validation (volume check)
2. Try NN-guided beam search (fast, may miss solutions)
3. If NN fails, fall back to DLX (exhaustive, guaranteed correct)
4. Return solution or None
"""

import time
import os
import sys
import multiprocessing as mp
import random
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1.solver import solve as dlx_solve, cube_root_int
from phase2.nn_solver import (
    nn_solve,
    dfs_solve_from_frontier,
    complete_solve_ordered,
    complete_solve_from_frontier,
)
from phase2.search_profiles import (
    resolve_search_profile,
    resolve_runtime_search_settings,
    resolve_retry_search_settings,
    resolve_structural_fallback_settings,
)
from phase2.train import load_model


def _solution_from_preplaced_input(pieces, grid_size):
    """Return a solution if pieces are already absolute, disjoint in-grid placements."""
    expected_volume = grid_size ** 3
    if sum(len(p) for p in pieces) != expected_volume:
        return None

    used = set()
    solution = {}
    for pidx, piece in enumerate(pieces):
        cells = set()
        for cell in piece:
            if len(cell) != 3:
                return None
            x, y, z = int(cell[0]), int(cell[1]), int(cell[2])
            if not (0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size):
                return None
            cells.add((x, y, z))
        if len(cells) != len(piece):
            return None
        if used & cells:
            return None
        used |= cells
        solution[pidx] = frozenset(cells)

    if len(used) != expected_volume:
        return None
    return solution


def _frontier_state_key(state_dict):
    occupied = frozenset(tuple(cell) for cell in state_dict.get('occupied', ()))
    remaining = tuple(sorted(int(i) for i in state_dict.get('remaining_indices', ())))
    return occupied, remaining


def _merge_frontier_sources(frontier_sources, max_states=8, per_source_cap=2):
    """Combine frontier roots from multiple search stages without losing source coverage."""
    if max_states is None or max_states <= 0:
        max_states = 8
    if per_source_cap is None or per_source_cap <= 0:
        per_source_cap = 1

    selected = []
    seen = set()
    overflow = []

    normalized = []
    for source_name, states in frontier_sources:
        ordered = sorted(states, key=lambda s: float(s.get("score", 0.0)), reverse=True)
        normalized.append((source_name, ordered))

    for source_name, states in normalized:
        kept_here = 0
        for state in states:
            key = _frontier_state_key(state)
            if key in seen:
                continue
            if kept_here < per_source_cap and len(selected) < max_states:
                selected.append(state)
                seen.add(key)
                kept_here += 1
            else:
                overflow.append(state)

    overflow.sort(key=lambda s: float(s.get("score", 0.0)), reverse=True)
    for state in overflow:
        if len(selected) >= max_states:
            break
        key = _frontier_state_key(state)
        if key in seen:
            continue
        selected.append(state)
        seen.add(key)

    return selected


def _block_size_profiles_125():
    """All count triples (n3,n4,n5) such that 3*n3 + 4*n4 + 5*n5 = 125."""
    return _block_size_profiles(125)


@lru_cache(maxsize=16)
def _block_size_profiles(volume):
    """All count triples (n3,n4,n5) such that 3*n3 + 4*n4 + 5*n5 = volume."""
    profiles = []
    for n3 in range(0, volume // 3 + 1):
        for n4 in range(0, (volume - 3 * n3) // 4 + 1):
            rem = volume - 3 * n3 - 4 * n4
            if rem < 0 or rem % 5 != 0:
                continue
            n5 = rem // 5
            profiles.append((n3, n4, n5))
    return profiles


def _bounds_feasible(r3, r4, r5, blocks_left, minmax):
    if blocks_left == 0:
        return r3 == 0 and r4 == 0 and r5 == 0
    (min3, max3), (min4, max4), (min5, max5) = minmax
    return (
        min3 * blocks_left <= r3 <= max3 * blocks_left and
        min4 * blocks_left <= r4 <= max4 * blocks_left and
        min5 * blocks_left <= r5 <= max5 * blocks_left
    )


def _allocate_block_profiles_125(num_blocks, c3, c4, c5):
    """Greedy profile allocation for block volume 125."""
    return _allocate_block_profiles(num_blocks, c3, c4, c5, 125)


def _allocate_block_profiles(num_blocks, c3, c4, c5, volume):
    """Greedy profile allocation for blocks of given volume."""
    profiles = _block_size_profiles(volume)
    minmax = (
        (min(p[0] for p in profiles), max(p[0] for p in profiles)),
        (min(p[1] for p in profiles), max(p[1] for p in profiles)),
        (min(p[2] for p in profiles), max(p[2] for p in profiles)),
    )

    out = []
    rem3, rem4, rem5 = c3, c4, c5
    for i in range(num_blocks):
        blocks_left = num_blocks - i
        # Dynamic per-block targets from remaining counts.
        t3 = rem3 / blocks_left
        t4 = rem4 / blocks_left
        t5 = rem5 / blocks_left

        candidates = []
        for p3, p4, p5 in profiles:
            if p3 > rem3 or p4 > rem4 or p5 > rem5:
                continue
            r3 = rem3 - p3
            r4 = rem4 - p4
            r5 = rem5 - p5
            if not _bounds_feasible(r3, r4, r5, blocks_left - 1, minmax):
                continue
            # Prefer profiles close to remaining ratio.
            score = abs(p3 - t3) + abs(p4 - t4) + abs(p5 - t5)
            candidates.append((score, (p3, p4, p5)))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        p3, p4, p5 = candidates[0][1]
        out.append((p3, p4, p5))
        rem3 -= p3
        rem4 -= p4
        rem5 -= p5

    if rem3 != 0 or rem4 != 0 or rem5 != 0:
        return None
    return out


def _solve_blockwise_5cube(
    pieces,
    grid_size,
    device='cpu',
    block_timeout_dlx=8.0,
    block_timeout_nn=0.0,
    trials=3,
    retries_per_block=3,
):
    """Solve large cubes by decomposing into independent 5x5x5 exact subproblems."""
    if grid_size < 10 or grid_size % 5 != 0:
        return None, {
            'reason': 'unsupported_grid',
            'blocks_total': 0,
            'blocks_solved': 0,
        }

    size_to_indices = {3: [], 4: [], 5: []}
    for idx, piece in enumerate(pieces):
        s = len(piece)
        if s not in size_to_indices:
            return None, {
                'reason': f'unsupported_piece_size_{s}',
                'blocks_total': 0,
                'blocks_solved': 0,
            }
        size_to_indices[s].append(idx)

    c3, c4, c5 = len(size_to_indices[3]), len(size_to_indices[4]), len(size_to_indices[5])
    if 3 * c3 + 4 * c4 + 5 * c5 != grid_size ** 3:
        return None, {
            'reason': 'volume_mismatch',
            'blocks_total': 0,
            'blocks_solved': 0,
        }

    nblk = grid_size // 5
    num_blocks = nblk ** 3
    profiles = _allocate_block_profiles_125(num_blocks, c3, c4, c5)
    if profiles is None:
        return None, {
            'reason': 'profile_allocation_failed',
            'blocks_total': num_blocks,
            'blocks_solved': 0,
        }

    block_coords = [
        (bx, by, bz)
        for bx in range(nblk)
        for by in range(nblk)
        for bz in range(nblk)
    ]

    best_fail = {
        'reason': 'block_unsolved',
        'blocks_total': num_blocks,
        'blocks_solved': 0,
        'failed_block': None,
        'trial': 0,
    }
    base_seed = grid_size * 1009 + len(pieces) * 9173

    for trial in range(max(1, int(trials))):
        rng = random.Random(base_seed + trial)
        pools = {
            3: list(size_to_indices[3]),
            4: list(size_to_indices[4]),
            5: list(size_to_indices[5]),
        }
        rng.shuffle(pools[3])
        rng.shuffle(pools[4])
        rng.shuffle(pools[5])

        global_solution = {}
        solved_blocks = 0
        fail_info = None

        for block_id, (bx, by, bz) in enumerate(block_coords):
            p3, p4, p5 = profiles[block_id]
            block_solved = False

            for retry in range(max(1, int(retries_per_block))):
                if len(pools[3]) < p3 or len(pools[4]) < p4 or len(pools[5]) < p5:
                    fail_info = {
                        'reason': 'pool_underflow',
                        'blocks_total': num_blocks,
                        'blocks_solved': solved_blocks,
                        'failed_block': (bx, by, bz),
                        'trial': trial,
                    }
                    break

                if retry == 0:
                    sel3 = pools[3][:p3]
                    sel4 = pools[4][:p4]
                    sel5 = pools[5][:p5]
                else:
                    sel3 = rng.sample(pools[3], p3) if p3 > 0 else []
                    sel4 = rng.sample(pools[4], p4) if p4 > 0 else []
                    sel5 = rng.sample(pools[5], p5) if p5 > 0 else []
                idxs = sel3 + sel4 + sel5
                block_pieces = [pieces[i] for i in idxs]

                block_res = hybrid_solve(
                    block_pieces,
                    grid_size=5,
                    model_name=None,
                    beam_width=0,
                    timeout_nn=block_timeout_nn,
                    timeout_dlx=block_timeout_dlx,
                    device=device,
                    verbose=False,
                )
                block_sol = block_res.get('solution')
                if block_sol is None:
                    continue

                ox, oy, oz = 5 * bx, 5 * by, 5 * bz
                for local_pidx, cells in block_sol.items():
                    global_idx = idxs[local_pidx]
                    shifted = frozenset((x + ox, y + oy, z + oz) for x, y, z in cells)
                    global_solution[global_idx] = shifted

                # Commit chosen pieces to this block.
                for i in sel3:
                    pools[3].remove(i)
                for i in sel4:
                    pools[4].remove(i)
                for i in sel5:
                    pools[5].remove(i)
                solved_blocks += 1
                block_solved = True
                break

            if fail_info is not None:
                break
            if not block_solved:
                fail_info = {
                    'reason': 'block_unsolved',
                    'blocks_total': num_blocks,
                    'blocks_solved': solved_blocks,
                    'failed_block': (bx, by, bz),
                    'trial': trial,
                }
                break

        if fail_info is None:
            if len(global_solution) != len(pieces):
                fail_info = {
                    'reason': 'incomplete_global_solution',
                    'blocks_total': num_blocks,
                    'blocks_solved': solved_blocks,
                    'failed_block': None,
                    'trial': trial,
                }
            else:
                return global_solution, {
                    'reason': None,
                    'blocks_total': num_blocks,
                    'blocks_solved': solved_blocks,
                    'trial': trial,
                }

        if fail_info.get('blocks_solved', 0) > best_fail.get('blocks_solved', 0):
            best_fail = fail_info

    best_fail['reason'] = f"block_planner_failed:{best_fail.get('reason')}"
    return None, best_fail


def _find_block_sizes(grid_size, min_block=5, max_block_vol=512):
    """Find valid block sizes for a grid, sorted smallest-first.

    A valid block size d satisfies:
      - grid_size % d == 0
      - d >= min_block  (pieces of size 5 need at least extent 5)
      - d³ <= max_block_vol  (keep sub-problems DLX-tractable)
      - d < grid_size  (decomposing into 1 block is pointless)
    """
    candidates = []
    for d in range(min_block, grid_size):
        if grid_size % d == 0 and d ** 3 <= max_block_vol:
            candidates.append(d)
    return candidates


def _solve_blockwise_general(
    pieces,
    grid_size,
    block_size,
    device='cpu',
    block_timeout_dlx=8.0,
    block_timeout_nn=0.0,
    trials=3,
    retries_per_block=3,
):
    """Solve large cubes by decomposing into independent block_size³ sub-problems."""
    if grid_size % block_size != 0 or block_size < 5:
        return None, {'reason': 'unsupported_grid', 'blocks_total': 0, 'blocks_solved': 0}

    block_volume = block_size ** 3

    size_to_indices = {3: [], 4: [], 5: []}
    for idx, piece in enumerate(pieces):
        s = len(piece)
        if s not in size_to_indices:
            return None, {'reason': f'unsupported_piece_size_{s}',
                          'blocks_total': 0, 'blocks_solved': 0}
        size_to_indices[s].append(idx)

    c3, c4, c5 = len(size_to_indices[3]), len(size_to_indices[4]), len(size_to_indices[5])
    if 3 * c3 + 4 * c4 + 5 * c5 != grid_size ** 3:
        return None, {'reason': 'volume_mismatch', 'blocks_total': 0, 'blocks_solved': 0}

    nblk = grid_size // block_size
    num_blocks = nblk ** 3
    profiles = _allocate_block_profiles(num_blocks, c3, c4, c5, block_volume)
    if profiles is None:
        return None, {'reason': 'profile_allocation_failed',
                      'blocks_total': num_blocks, 'blocks_solved': 0}

    block_coords = [
        (bx, by, bz)
        for bx in range(nblk)
        for by in range(nblk)
        for bz in range(nblk)
    ]

    best_fail = {
        'reason': 'block_unsolved',
        'blocks_total': num_blocks,
        'blocks_solved': 0,
        'failed_block': None,
        'trial': 0,
    }
    base_seed = grid_size * 1009 + len(pieces) * 9173 + block_size * 7

    for trial in range(max(1, int(trials))):
        rng = random.Random(base_seed + trial)
        pools = {
            3: list(size_to_indices[3]),
            4: list(size_to_indices[4]),
            5: list(size_to_indices[5]),
        }
        rng.shuffle(pools[3])
        rng.shuffle(pools[4])
        rng.shuffle(pools[5])

        global_solution = {}
        solved_blocks = 0
        fail_info = None

        for block_id, (bx, by, bz) in enumerate(block_coords):
            p3, p4, p5 = profiles[block_id]
            block_solved = False

            for retry in range(max(1, int(retries_per_block))):
                if len(pools[3]) < p3 or len(pools[4]) < p4 or len(pools[5]) < p5:
                    fail_info = {
                        'reason': 'pool_underflow',
                        'blocks_total': num_blocks,
                        'blocks_solved': solved_blocks,
                        'failed_block': (bx, by, bz),
                        'trial': trial,
                    }
                    break

                if retry == 0:
                    sel3 = pools[3][:p3]
                    sel4 = pools[4][:p4]
                    sel5 = pools[5][:p5]
                else:
                    sel3 = rng.sample(pools[3], p3) if p3 > 0 else []
                    sel4 = rng.sample(pools[4], p4) if p4 > 0 else []
                    sel5 = rng.sample(pools[5], p5) if p5 > 0 else []
                idxs = sel3 + sel4 + sel5
                block_pieces = [pieces[i] for i in idxs]

                block_res = hybrid_solve(
                    block_pieces,
                    grid_size=block_size,
                    model_name=None,
                    beam_width=0,
                    timeout_nn=block_timeout_nn,
                    timeout_dlx=block_timeout_dlx,
                    device=device,
                    verbose=False,
                )
                block_sol = block_res.get('solution')
                if block_sol is None:
                    continue

                ox, oy, oz = block_size * bx, block_size * by, block_size * bz
                for local_pidx, cells in block_sol.items():
                    global_idx = idxs[local_pidx]
                    shifted = frozenset((x + ox, y + oy, z + oz) for x, y, z in cells)
                    global_solution[global_idx] = shifted

                for i in sel3:
                    pools[3].remove(i)
                for i in sel4:
                    pools[4].remove(i)
                for i in sel5:
                    pools[5].remove(i)
                solved_blocks += 1
                block_solved = True
                break

            if fail_info is not None:
                break
            if not block_solved:
                fail_info = {
                    'reason': 'block_unsolved',
                    'blocks_total': num_blocks,
                    'blocks_solved': solved_blocks,
                    'failed_block': (bx, by, bz),
                    'trial': trial,
                }
                break

        if fail_info is None:
            if len(global_solution) != len(pieces):
                fail_info = {
                    'reason': 'incomplete_global_solution',
                    'blocks_total': num_blocks,
                    'blocks_solved': solved_blocks,
                    'failed_block': None,
                    'trial': trial,
                }
            else:
                return global_solution, {
                    'reason': None,
                    'blocks_total': num_blocks,
                    'blocks_solved': solved_blocks,
                    'block_size': block_size,
                    'trial': trial,
                }

        if fail_info.get('blocks_solved', 0) > best_fail.get('blocks_solved', 0):
            best_fail = fail_info

    best_fail['reason'] = f"block_planner_failed:{best_fail.get('reason')}"
    best_fail['block_size'] = block_size
    return None, best_fail


def _dlx_worker(pieces, grid_size, out_q):
    """Worker process entrypoint for timeout-safe DLX solving."""
    try:
        sols = dlx_solve(pieces, grid_size=grid_size, find_all=False)
        out_q.put(("ok", sols))
    except Exception as exc:  # pragma: no cover - defensive path
        out_q.put(("err", repr(exc)))


def _run_dlx_with_timeout(pieces, grid_size, timeout_seconds):
    """Run DLX in a separate process and enforce a hard timeout.

    Returns:
        (solutions, timed_out, error_message)
    """
    if timeout_seconds is None or timeout_seconds <= 0:
        try:
            return dlx_solve(pieces, grid_size=grid_size, find_all=False), False, None
        except Exception as exc:  # pragma: no cover - defensive path
            return [], False, repr(exc)

    # multiprocessing spawn-based timeout is fragile in interactive contexts
    # (e.g., notebook cells, stdin/heredoc runners) where __main__.__file__ is absent.
    try:
        import __main__ as main_mod
        can_spawn_timeout = bool(getattr(main_mod, "__file__", None))
    except Exception:  # pragma: no cover - defensive path
        can_spawn_timeout = False

    # Prefer spawn when possible; otherwise try fork (works from interactive callers).
    contexts = ["spawn"] if can_spawn_timeout else []
    if not can_spawn_timeout:
        try:
            mp.get_context("fork")
            contexts.append("fork")
        except Exception:  # pragma: no cover - platform dependent
            pass

    for ctx_name in contexts:
        try:
            ctx = mp.get_context(ctx_name)
            out_q = ctx.Queue()
            proc = ctx.Process(target=_dlx_worker, args=(pieces, grid_size, out_q))
            proc.start()
            proc.join(timeout_seconds)

            if proc.is_alive():
                proc.terminate()
                proc.join()
                return [], True, None

            if out_q.empty():
                return [], False, "DLX worker exited without result"

            status, payload = out_q.get()
            if status == "ok":
                return payload, False, None
            return [], False, payload
        except Exception:
            continue

    # No timeout-enforced worker could be started. Refuse in-process DLX here
    # to avoid unbounded hangs on large instances.
    return [], True, "DLX timeout worker unavailable in this runtime"


def hybrid_solve(pieces, grid_size=None, model_name="soma_3x3x3",
                 beam_width=None, timeout_nn=30, timeout_dlx=120,
                 device='cpu', verbose=True,
                 max_candidates_per_state=None,
                 enable_pocket_pruning=None,
                 placement_ranker=None,
                 max_children_per_layer=None,
                 beam_diversity_slots=None,
                 beam_diversity_metric=None,
                 piece_branching_width=None,
                 piece_branching_slack=None,
                 nn_retry_on_fail=None,
                 retry_search_profile=None,
                 retry_beam_mult=1.5,
                 retry_timeout_mult=1.5,
                 retry_max_candidates_mult=1.5,
                 retry_max_children_mult=1.5,
                 retry_enable_pocket_pruning=None,
                 nn_frontier_dfs=None,
                 dfs_timeout=20.0,
                 dfs_max_frontier_states=16,
                 dfs_branch_limit=12,
                 dfs_max_nodes=15000,
                 dfs_placement_ranker='contact',
                 dfs_enable_pocket_pruning=None,
                 frontier_harvest_search_profile=None,
                 frontier_harvest_timeout=None,
                 frontier_portfolio_sources=None,
                 frontier_merge_sources=None,
                 frontier_merge_max_states=8,
                 frontier_merge_per_source_cap=2,
                 nn_frontier_complete=None,
                 frontier_complete_timeout=10.0,
                 frontier_complete_max_frontier_states=4,
                 frontier_complete_max_nodes=50000,
                 frontier_complete_placement_ranker='contact',
                 frontier_complete_enable_pocket_pruning=None,
                 frontier_complete_use_transposition=True,
                 nn_complete_fallback=None,
                 complete_timeout=60.0,
                 complete_max_nodes=200000,
                 complete_placement_ranker='contact',
                 complete_enable_pocket_pruning=None,
                 complete_use_transposition=True):
    """Solve polycube packing with NN-first, DLX-fallback strategy.

    Args:
        pieces: list of pieces (each a list of (x,y,z) tuples)
        grid_size: side length of target cube (auto-detected if None)
        model_name: name of saved NN model (None to skip NN)
        beam_width: beam search width for NN solver
        timeout_nn: max seconds for NN solver
        timeout_dlx: max seconds for DLX solver (only on Unix)
        device: 'cpu' or 'cuda'
        verbose: print progress
        max_candidates_per_state: max child states generated per beam node
        enable_pocket_pruning: prune disconnected empty-space pockets
        placement_ranker: placement ranking mode for NN truncation
        max_children_per_layer: optional cap on total generated children per depth
        beam_diversity_slots: reserve some beam slots for structurally distinct
            frontier states after value ranking
        beam_diversity_metric: coarse structural signature used for diversity
        piece_branching_width: number of near-MRV pieces expanded per beam state
        piece_branching_slack: alternate-piece MRV slack for multi-piece branching
        nn_retry_on_fail: if True, run a second NN pass before DLX.
            If None, enables a configured retry profile automatically.
        retry_beam_mult: multiplier for beam_width in retry pass
        retry_timeout_mult: multiplier for timeout_nn in retry pass
        retry_max_candidates_mult: multiplier for per-state candidate cap in retry
        retry_max_children_mult: multiplier for per-layer child cap in retry
        retry_enable_pocket_pruning: override pocket pruning for retry pass.
            If None, uses enable_pocket_pruning.
        nn_frontier_dfs: run bounded DFS from beam frontier before DLX
        dfs_timeout: timeout budget (seconds) for frontier DFS retry
        dfs_max_frontier_states: number of top frontier roots to explore
        dfs_branch_limit: max child placements explored per DFS node
        dfs_max_nodes: max DFS node expansions
        dfs_placement_ranker: placement ranking for DFS ('contact'/'policy'/'hybrid')
        dfs_enable_pocket_pruning: override pocket pruning for DFS pass.
            If None, uses enable_pocket_pruning.
        frontier_harvest_search_profile: optional wider search profile used to
            gather a richer frontier before frontier-complete search.
        frontier_harvest_timeout: timeout budget for the frontier-harvest pass.
        frontier_portfolio_sources: if True, run DFS/frontier-complete against
            retry and harvest frontier sources independently before DLX.
        frontier_merge_sources: if True, merge beam/retry/harvest frontier pools
            before DFS/frontier-complete instead of choosing only one source.
        frontier_merge_max_states: max number of merged frontier roots kept.
        frontier_merge_per_source_cap: minimum per-source allocation before
            filling the remainder by score.
        nn_frontier_complete: run complete deterministic search from saved
            frontier roots before DLX.
        frontier_complete_timeout: timeout budget (seconds) for frontier-complete pass.
        frontier_complete_max_frontier_states: number of frontier roots to explore.
        frontier_complete_max_nodes: max node expansions for frontier-complete pass.
        frontier_complete_placement_ranker: placement ranking for frontier-complete pass.
        frontier_complete_enable_pocket_pruning: override pocket pruning for
            frontier-complete pass. If None, uses enable_pocket_pruning.
        frontier_complete_use_transposition: enable transposition table for
            frontier-complete pass.
        nn_complete_fallback: run complete deterministic local search before DLX.
        complete_timeout: timeout budget (seconds) for complete fallback search.
        complete_max_nodes: max node expansions for complete fallback search.
        complete_placement_ranker: placement ranking for complete fallback
            ('contact'/'policy'/'hybrid').
        complete_enable_pocket_pruning: override pocket pruning for complete pass.
            If None, uses enable_pocket_pruning.
        complete_use_transposition: enable transposition table in complete pass.

    Returns:
        dict: {
            'solution': dict mapping piece_idx -> frozenset of cells, or None,
            'method': 'nn' | 'dlx' | None,
            'submethod': str (beam/retry/dfs_frontier/complete_fallback/dlx_exact/...)
            'time': float (seconds),
        }
    """
    t0 = time.time()

    # ── Volume check ──
    total_vol = sum(len(p) for p in pieces)
    if grid_size is None:
        grid_size = cube_root_int(total_vol)
        if grid_size is None:
            if verbose:
                print(f"Volume {total_vol} is not a perfect cube. No solution possible.")
            return {'solution': None, 'method': None, 'time': time.time() - t0}
    else:
        if total_vol != grid_size ** 3:
            if verbose:
                print(f"Volume mismatch: {total_vol} != {grid_size}^3")
            return {'solution': None, 'method': None, 'time': time.time() - t0}

    if verbose:
        print(f"Solving {grid_size}x{grid_size}x{grid_size} cube "
              f"({len(pieces)} pieces, {total_vol} cells)")

    runtime = resolve_runtime_search_settings(
        grid_size,
        beam_width=beam_width,
        max_candidates_per_state=max_candidates_per_state,
        placement_ranker=placement_ranker,
        enable_pocket_pruning=enable_pocket_pruning,
        max_children_per_layer=max_children_per_layer,
        beam_diversity_slots=beam_diversity_slots,
        beam_diversity_metric=beam_diversity_metric,
        piece_branching_width=piece_branching_width,
        piece_branching_slack=piece_branching_slack,
        default_beam_width=64,
        default_max_candidates_per_state=50,
    )
    beam_width = runtime["beam_width"]
    max_candidates_per_state = runtime["max_candidates_per_state"]
    placement_ranker = runtime["placement_ranker"]
    enable_pocket_pruning = runtime["enable_pocket_pruning"]
    max_children_per_layer = runtime["max_children_per_layer"]
    beam_diversity_slots = runtime["beam_diversity_slots"]
    beam_diversity_metric = runtime["beam_diversity_metric"]
    piece_branching_width = runtime["piece_branching_width"]
    piece_branching_slack = runtime["piece_branching_slack"]

    # ── Try NN solver first ──
    nn_solution = None
    nn_time = 0.0
    stages_attempted = []

    if model_name is not None:
        try:
            if verbose:
                print(f"\n[1] NN beam search (beam_width={beam_width}, "
                      f"timeout={timeout_nn}s)...")

            model, _, metadata = load_model(model_name, device=device)
            max_pieces = model.in_channels - 1

            structural = resolve_structural_fallback_settings(grid_size)
            if structural is not None:
                if nn_frontier_dfs is None:
                    nn_frontier_dfs = structural.get("nn_frontier_dfs", False)
                dfs_timeout = structural.get("dfs_timeout", dfs_timeout)
                dfs_max_frontier_states = structural.get("dfs_max_frontier_states", dfs_max_frontier_states)
                dfs_branch_limit = structural.get("dfs_branch_limit", dfs_branch_limit)
                dfs_max_nodes = structural.get("dfs_max_nodes", dfs_max_nodes)
                dfs_placement_ranker = structural.get("dfs_placement_ranker", dfs_placement_ranker)
                if dfs_enable_pocket_pruning is None:
                    dfs_enable_pocket_pruning = structural.get(
                        "dfs_enable_pocket_pruning",
                        dfs_enable_pocket_pruning,
                    )
                if nn_frontier_complete is None:
                    nn_frontier_complete = structural.get("nn_frontier_complete", False)
                if frontier_harvest_search_profile is None:
                    frontier_harvest_search_profile = structural.get(
                        "frontier_harvest_search_profile",
                        frontier_harvest_search_profile,
                    )
                if frontier_harvest_timeout is None:
                    frontier_harvest_timeout = structural.get(
                        "frontier_harvest_timeout",
                        frontier_harvest_timeout,
                    )
                if frontier_portfolio_sources is None:
                    frontier_portfolio_sources = structural.get(
                        "frontier_portfolio_sources",
                        frontier_portfolio_sources,
                    )
                if frontier_merge_sources is None:
                    frontier_merge_sources = structural.get(
                        "frontier_merge_sources",
                        frontier_merge_sources,
                    )
                frontier_merge_max_states = structural.get(
                    "frontier_merge_max_states",
                    frontier_merge_max_states,
                )
                frontier_merge_per_source_cap = structural.get(
                    "frontier_merge_per_source_cap",
                    frontier_merge_per_source_cap,
                )
                frontier_complete_timeout = structural.get(
                    "frontier_complete_timeout",
                    frontier_complete_timeout,
                )
                frontier_complete_max_frontier_states = structural.get(
                    "frontier_complete_max_frontier_states",
                    frontier_complete_max_frontier_states,
                )
                frontier_complete_max_nodes = structural.get(
                    "frontier_complete_max_nodes",
                    frontier_complete_max_nodes,
                )
                frontier_complete_placement_ranker = structural.get(
                    "frontier_complete_placement_ranker",
                    frontier_complete_placement_ranker,
                )
                if frontier_complete_enable_pocket_pruning is None:
                    frontier_complete_enable_pocket_pruning = structural.get(
                        "frontier_complete_enable_pocket_pruning",
                        frontier_complete_enable_pocket_pruning,
                    )
                frontier_complete_use_transposition = structural.get(
                    "frontier_complete_use_transposition",
                    frontier_complete_use_transposition,
                )
            if nn_frontier_dfs is None:
                nn_frontier_dfs = False
            if nn_frontier_complete is None:
                nn_frontier_complete = False
            if frontier_portfolio_sources is None:
                frontier_portfolio_sources = False
            if frontier_merge_sources is None:
                frontier_merge_sources = False
            if nn_complete_fallback is None:
                nn_complete_fallback = False

            beam_frontier = []
            t_nn = time.time()
            stages_attempted.append('beam')
            nn_solution = nn_solve(
                pieces, grid_size, model,
                max_pieces=max_pieces, beam_width=beam_width,
                timeout=timeout_nn, device=device,
                max_candidates_per_state=max_candidates_per_state,
                enable_pocket_pruning=enable_pocket_pruning,
                placement_ranker=placement_ranker,
                max_children_per_layer=max_children_per_layer,
                beam_diversity_slots=beam_diversity_slots,
                beam_diversity_metric=beam_diversity_metric,
                piece_branching_width=piece_branching_width,
                piece_branching_slack=piece_branching_slack,
                frontier_out=beam_frontier,
            )
            nn_time = time.time() - t_nn

            if nn_solution is not None:
                if verbose:
                    print(f"    NN solved in {nn_time:.3f}s!")
                return {
                    'solution': nn_solution,
                    'method': 'nn',
                    'submethod': 'beam',
                    'time': nn_time,
                    'stages_attempted': stages_attempted,
                }

            auto_retry_enabled = resolve_retry_search_settings(
                grid_size,
                profile_name=retry_search_profile,
            ) is not None
            use_retry = auto_retry_enabled if nn_retry_on_fail is None else bool(nn_retry_on_fail)

            if use_retry:
                retry_frontier = []
                retry_timeout = max(timeout_nn, timeout_nn * retry_timeout_mult)
                retry_profile = resolve_retry_search_settings(
                    grid_size,
                    profile_name=retry_search_profile,
                )

                if retry_profile is not None:
                    retry_beam_width = retry_profile["beam_width"]
                    retry_max_candidates = retry_profile["max_candidates_per_state"]
                    retry_max_children = retry_profile["max_children_per_layer"]
                    retry_ranker = retry_profile["placement_ranker"]
                    retry_pocket_pruning = retry_profile["enable_pocket_pruning"]
                    retry_beam_diversity_slots = retry_profile["beam_diversity_slots"]
                    retry_beam_diversity_metric = retry_profile["beam_diversity_metric"]
                    retry_piece_branching_width = retry_profile["piece_branching_width"]
                    retry_piece_branching_slack = retry_profile["piece_branching_slack"]
                else:
                    retry_beam_width = max(beam_width + 1, int(round(beam_width * retry_beam_mult)))
                    retry_max_candidates = max(
                        max_candidates_per_state + 1,
                        int(round(max_candidates_per_state * retry_max_candidates_mult))
                    )
                    retry_max_children = None
                    if max_children_per_layer is not None:
                        retry_max_children = max(
                            max_children_per_layer + 1,
                            int(round(max_children_per_layer * retry_max_children_mult))
                        )
                    retry_ranker = placement_ranker
                    retry_pocket_pruning = enable_pocket_pruning
                    retry_beam_diversity_slots = beam_diversity_slots
                    retry_beam_diversity_metric = beam_diversity_metric
                    retry_piece_branching_width = piece_branching_width
                    retry_piece_branching_slack = piece_branching_slack

                if retry_enable_pocket_pruning is not None:
                    retry_pocket_pruning = bool(retry_enable_pocket_pruning)

                if verbose:
                    if retry_profile is not None:
                        print(
                            "    NN first pass failed; retrying profile "
                            f"{retry_profile['profile_name']} "
                            f"(beam_width={retry_beam_width}, timeout={retry_timeout}s)..."
                        )
                    else:
                        print(
                            "    NN first pass failed; retrying widened pass "
                            f"(beam_width={retry_beam_width}, timeout={retry_timeout}s)..."
                        )

                t_retry = time.time()
                stages_attempted.append(
                    retry_profile['profile_name'] if retry_profile is not None else 'retry_beam'
                )
                nn_solution = nn_solve(
                    pieces, grid_size, model,
                    max_pieces=max_pieces, beam_width=retry_beam_width,
                    timeout=retry_timeout, device=device,
                    max_candidates_per_state=retry_max_candidates,
                    enable_pocket_pruning=retry_pocket_pruning,
                    placement_ranker=retry_ranker,
                    max_children_per_layer=retry_max_children,
                    beam_diversity_slots=retry_beam_diversity_slots,
                    beam_diversity_metric=retry_beam_diversity_metric,
                    piece_branching_width=retry_piece_branching_width,
                    piece_branching_slack=retry_piece_branching_slack,
                    frontier_out=retry_frontier,
                )
                nn_time += time.time() - t_retry

                if nn_solution is not None:
                    if verbose:
                        print(f"    NN solved on retry in {nn_time:.3f}s total!")
                    return {
                        'solution': nn_solution,
                        'method': 'nn',
                        'submethod': 'retry_beam',
                        'time': nn_time,
                        'stages_attempted': stages_attempted,
                    }
            else:
                retry_frontier = []

            harvest_frontier = []
            frontier_for_structural = retry_frontier if retry_frontier else beam_frontier

            if frontier_harvest_search_profile not in (None, "__disable__"):
                harvest_profile = resolve_search_profile(frontier_harvest_search_profile)
                harvest_timeout = frontier_harvest_timeout
                if harvest_timeout is None:
                    harvest_timeout = max(timeout_nn, 8.0)
                harvest_frontier = []

                if verbose:
                    print(
                        "    Running frontier-harvest profile "
                        f"{frontier_harvest_search_profile} "
                        f"(beam_width={harvest_profile['beam_width']}, "
                        f"timeout={harvest_timeout}s)..."
                    )

                t_harvest = time.time()
                stages_attempted.append(frontier_harvest_search_profile)
                harvest_solution = nn_solve(
                    pieces, grid_size, model,
                    max_pieces=max_pieces,
                    beam_width=harvest_profile["beam_width"],
                    timeout=harvest_timeout,
                    device=device,
                    max_candidates_per_state=harvest_profile["max_candidates_per_state"],
                    enable_pocket_pruning=harvest_profile["enable_pocket_pruning"],
                    placement_ranker=harvest_profile["placement_ranker"],
                    max_children_per_layer=harvest_profile["max_children_per_layer"],
                    beam_diversity_slots=harvest_profile["beam_diversity_slots"],
                    beam_diversity_metric=harvest_profile["beam_diversity_metric"],
                    piece_branching_width=harvest_profile["piece_branching_width"],
                    piece_branching_slack=harvest_profile["piece_branching_slack"],
                    frontier_out=harvest_frontier,
                )
                nn_time += time.time() - t_harvest

                if harvest_solution is not None:
                    if verbose:
                        print(f"    Frontier-harvest beam solved in {nn_time:.3f}s total!")
                    return {
                        'solution': harvest_solution,
                        'method': 'nn',
                        'submethod': 'frontier_harvest',
                        'time': nn_time,
                        'stages_attempted': stages_attempted,
                    }

                if harvest_frontier:
                    frontier_for_structural = harvest_frontier

            structural_frontiers = []
            if frontier_portfolio_sources:
                source_signatures = set()
                for source_name, frontier_states in (
                    ("retry", retry_frontier),
                    ("harvest", harvest_frontier),
                    ("beam", beam_frontier),
                ):
                    if not frontier_states:
                        continue
                    signature = tuple(
                        _frontier_state_key(state)
                        for state in frontier_states[:min(3, len(frontier_states))]
                    )
                    if signature in source_signatures:
                        continue
                    source_signatures.add(signature)
                    structural_frontiers.append((source_name, frontier_states))
            else:
                if frontier_merge_sources:
                    frontier_for_structural = _merge_frontier_sources(
                        [
                            ("beam", beam_frontier),
                            ("retry", retry_frontier),
                            ("harvest", harvest_frontier),
                        ],
                        max_states=frontier_merge_max_states,
                        per_source_cap=frontier_merge_per_source_cap,
                    )
                structural_frontiers = [("selected", frontier_for_structural)] if frontier_for_structural else []

            for source_name, structural_frontier in structural_frontiers:
                if nn_frontier_dfs and structural_frontier:
                    dfs_pocket_pruning = enable_pocket_pruning
                    if dfs_enable_pocket_pruning is not None:
                        dfs_pocket_pruning = bool(dfs_enable_pocket_pruning)

                    if verbose:
                        print(
                            "    Running frontier DFS retry "
                            f"[{source_name}] "
                            f"(roots={min(len(structural_frontier), dfs_max_frontier_states)}, "
                            f"timeout={dfs_timeout}s)..."
                        )

                    t_dfs = time.time()
                    stage_name = 'dfs_frontier' if source_name == 'selected' else f'dfs_frontier_{source_name}'
                    stages_attempted.append(stage_name)
                    dfs_solution, dfs_diag = dfs_solve_from_frontier(
                        pieces=pieces,
                        grid_size=grid_size,
                        model=model,
                        frontier_states=structural_frontier,
                        max_pieces=max_pieces,
                        timeout=dfs_timeout,
                        max_frontier_states=dfs_max_frontier_states,
                        branch_limit=dfs_branch_limit,
                        max_nodes=dfs_max_nodes,
                        device=device,
                        placement_ranker=dfs_placement_ranker,
                        enable_pocket_pruning=dfs_pocket_pruning,
                    )
                    nn_time += time.time() - t_dfs

                    if dfs_solution is not None:
                        if verbose:
                            print(f"    Frontier DFS solved in {nn_time:.3f}s total!")
                        return {
                            'solution': dfs_solution,
                            'method': 'nn',
                            'submethod': stage_name,
                            'time': nn_time,
                            'stages_attempted': stages_attempted,
                        }

                if nn_frontier_complete and structural_frontier:
                    complete_frontier_pocket_pruning = enable_pocket_pruning
                    if frontier_complete_enable_pocket_pruning is not None:
                        complete_frontier_pocket_pruning = bool(
                            frontier_complete_enable_pocket_pruning
                        )

                    if verbose:
                        print(
                            "    Running frontier-complete search "
                            f"[{source_name}] "
                            f"(roots={min(len(structural_frontier), frontier_complete_max_frontier_states)}, "
                            f"timeout={frontier_complete_timeout}s, "
                            f"max_nodes={frontier_complete_max_nodes})..."
                        )

                    t_complete_frontier = time.time()
                    stage_name = (
                        'frontier_complete'
                        if source_name == 'selected' else f'frontier_complete_{source_name}'
                    )
                    stages_attempted.append(stage_name)
                    complete_frontier_solution, complete_frontier_diag = complete_solve_from_frontier(
                        pieces=pieces,
                        grid_size=grid_size,
                        model=model,
                        frontier_states=structural_frontier,
                        max_pieces=max_pieces,
                        timeout=frontier_complete_timeout,
                        max_frontier_states=frontier_complete_max_frontier_states,
                        max_nodes=frontier_complete_max_nodes,
                        device=device,
                        placement_ranker=frontier_complete_placement_ranker,
                        enable_pocket_pruning=complete_frontier_pocket_pruning,
                        use_transposition=frontier_complete_use_transposition,
                    )
                    nn_time += time.time() - t_complete_frontier

                    if complete_frontier_solution is not None:
                        if verbose:
                            print(
                                f"    Frontier-complete search solved in {nn_time:.3f}s total!"
                            )
                        return {
                            'solution': complete_frontier_solution,
                            'method': 'nn',
                            'submethod': stage_name,
                            'time': nn_time,
                            'stages_attempted': stages_attempted,
                        }

            if nn_complete_fallback:
                complete_pocket_pruning = enable_pocket_pruning
                if complete_enable_pocket_pruning is not None:
                    complete_pocket_pruning = bool(complete_enable_pocket_pruning)

                if verbose:
                    print(
                        "    Running complete deterministic fallback "
                        f"(timeout={complete_timeout}s, max_nodes={complete_max_nodes})..."
                    )

                t_complete = time.time()
                stages_attempted.append('complete_fallback')
                complete_solution, complete_diag = complete_solve_ordered(
                    pieces=pieces,
                    grid_size=grid_size,
                    model=model,
                    max_pieces=max_pieces,
                    timeout=complete_timeout,
                    max_nodes=complete_max_nodes,
                    device=device,
                    placement_ranker=complete_placement_ranker,
                    enable_pocket_pruning=complete_pocket_pruning,
                    use_transposition=complete_use_transposition,
                )
                nn_time += time.time() - t_complete

                if complete_solution is not None:
                    if verbose:
                        print(f"    Complete fallback solved in {nn_time:.3f}s total!")
                        return {
                            'solution': complete_solution,
                            'method': 'nn',
                            'submethod': 'complete_fallback',
                            'time': nn_time,
                            'stages_attempted': stages_attempted,
                        }

                if verbose:
                    print(
                        "    Complete fallback failed "
                        f"(reason={complete_diag.get('failure_reason')}, "
                        f"elapsed={complete_diag.get('elapsed_sec', 0.0):.3f}s)."
                    )

            if verbose:
                print(f"    NN failed after {nn_time:.3f}s, falling back to DLX...")

        except FileNotFoundError:
            if verbose:
                print(f"    No trained model '{model_name}' found, skipping NN solver.")
        except Exception as e:
            if verbose:
                print(f"    NN solver error: {e}, falling back to DLX...")

    # ── Fall back to DLX ──
    if verbose:
        print(f"\n[2] DLX exact cover solver...")

    t_dlx = time.time()
    dlx_solutions, dlx_timed_out, dlx_error = _run_dlx_with_timeout(
        pieces, grid_size, timeout_dlx
    )
    dlx_time = time.time() - t_dlx

    if dlx_timed_out:
        if verbose:
            print(f"    DLX timed out after {timeout_dlx}s")
        return {
            'solution': None,
            'method': None,
            'submethod': 'dlx_timeout',
            'time': nn_time + dlx_time,
            'stages_attempted': stages_attempted + ['dlx'],
        }

    if dlx_error is not None:
        if verbose:
            print(f"    DLX error: {dlx_error}")
        return {
            'solution': None,
            'method': None,
            'submethod': 'dlx_error',
            'time': nn_time + dlx_time,
            'stages_attempted': stages_attempted + ['dlx'],
        }

    if dlx_solutions:
        if verbose:
            print(f"    DLX solved in {dlx_time:.3f}s")
        return {
            'solution': dlx_solutions[0],
            'method': 'dlx',
            'submethod': 'dlx_exact',
            'time': nn_time + dlx_time,
            'stages_attempted': stages_attempted + ['dlx'],
        }
    else:
        if verbose:
            print(f"    DLX found no solution ({dlx_time:.3f}s)")
        return {
            'solution': None,
            'method': None,
            'submethod': 'dlx_no_solution',
            'time': nn_time + dlx_time,
            'stages_attempted': stages_attempted + ['dlx'],
        }


def solve_size_gated(
    pieces,
    grid_size=None,
    model_name="soma_3x3x3",
    beam_width=None,
    timeout_nn=30,
    timeout_dlx=120,
    device='cpu',
    verbose=True,
    exact_only_max_grid=4,
    exact_first_max_grid=6,
    exact_first_timeout=30.0,
    large_allow_dlx=False,
    allow_preplaced_fastpath=True,
    block_planner_enabled=True,
    block_timeout_dlx=8.0,
    block_timeout_nn=0.0,
    block_planner_trials=3,
    block_retries_per_block=3,
    **hybrid_kwargs,
):
    """Size-gated orchestrator for staged scaling experiments.

    Routing:
    - grid <= exact_only_max_grid: exact-only (DLX)
    - exact_only_max_grid < grid <= exact_first_max_grid: exact-first, then hybrid
    - grid > exact_first_max_grid: hybrid (planner placeholder tier)

    Args:
        large_allow_dlx: allow large-tier DLX fallback when no model is present.
        allow_preplaced_fastpath: if True, return immediately when input pieces
            are already an absolute non-overlapping cube cover.
        block_planner_enabled: try 5x5x5 block decomposition on large grids.
        block_timeout_dlx: exact-solver timeout per 5x5x5 block.
        block_timeout_nn: nn timeout per 5x5x5 block.
        block_planner_trials: number of global reassignment attempts.
        block_retries_per_block: candidate resampling attempts per block.
    """
    t0 = time.time()
    total_vol = sum(len(p) for p in pieces)
    if grid_size is None:
        grid_size = cube_root_int(total_vol)
        if grid_size is None:
            return {
                'solution': None,
                'method': None,
                'submethod': 'invalid_volume',
                'controller': 'size_gated',
                'tier': 'invalid',
                'time': time.time() - t0,
            }
    elif total_vol != grid_size ** 3:
        return {
            'solution': None,
            'method': None,
            'submethod': 'volume_mismatch',
            'controller': 'size_gated',
            'tier': 'invalid',
            'time': time.time() - t0,
        }

    if allow_preplaced_fastpath:
        # Optional shortcut for callers that intentionally pass absolute tilings.
        preplaced_solution = _solution_from_preplaced_input(pieces, grid_size)
        if preplaced_solution is not None:
            return {
                'solution': preplaced_solution,
                'method': 'planner',
                'submethod': 'preplaced_input_cover',
                'controller': 'size_gated',
                'tier': 'preplaced_fastpath',
            'time': time.time() - t0,
        }

    planner_diag = None

    # Block decomposition planners (large grids only).
    if grid_size > exact_first_max_grid:
        if block_planner_enabled:
            # Try all valid block sizes, smallest first (smallest = fastest sub-problems).
            # max_block_vol=512 means block_size up to 8 (8³=512).
            block_sizes = _find_block_sizes(grid_size, min_block=5, max_block_vol=512)
            for bs in block_sizes:
                # Scale timeout with block volume: larger blocks need more time.
                scale = max(1.0, (bs ** 3) / 125.0)
                bs_timeout = block_timeout_dlx * scale
                # More pieces → more trials needed to find a compatible assignment.
                adaptive_trials = max(block_planner_trials,
                                      min(10, len(pieces) // 50))
                adaptive_retries = max(block_retries_per_block,
                                       min(8, len(pieces) // 80))
                block_solution, block_diag = _solve_blockwise_general(
                    pieces=pieces,
                    grid_size=grid_size,
                    block_size=bs,
                    device=device,
                    block_timeout_dlx=bs_timeout,
                    block_timeout_nn=block_timeout_nn,
                    trials=adaptive_trials,
                    retries_per_block=adaptive_retries,
                )
                if block_solution is not None:
                    return {
                        'solution': block_solution,
                        'method': 'planner',
                        'submethod': f'blockwise_{bs}cube',
                        'controller': 'size_gated',
                        'tier': 'large_planner',
                        'time': time.time() - t0,
                        'planner_diag': block_diag,
                    }
                planner_diag = block_diag

    if grid_size <= exact_only_max_grid:
        sols, timed_out, err = _run_dlx_with_timeout(pieces, grid_size, timeout_dlx)
        elapsed = time.time() - t0
        if timed_out:
            return {
                'solution': None,
                'method': None,
                'submethod': 'dlx_timeout',
                'controller': 'size_gated',
                'tier': 'small_exact',
                'time': elapsed,
            }
        if err is not None:
            return {
                'solution': None,
                'method': None,
                'submethod': 'dlx_error',
                'controller': 'size_gated',
                'tier': 'small_exact',
                'time': elapsed,
            }
        if sols:
            return {
                'solution': sols[0],
                'method': 'dlx',
                'submethod': 'dlx_exact',
                'controller': 'size_gated',
                'tier': 'small_exact',
                'time': elapsed,
            }
        return {
            'solution': None,
            'method': None,
            'submethod': 'dlx_no_solution',
            'controller': 'size_gated',
            'tier': 'small_exact',
            'time': elapsed,
        }

    if grid_size <= exact_first_max_grid:
        exact_sols, exact_timed_out, exact_err = _run_dlx_with_timeout(
            pieces, grid_size, exact_first_timeout
        )
        if exact_sols:
            return {
                'solution': exact_sols[0],
                'method': 'dlx',
                'submethod': 'dlx_exact_first',
                'controller': 'size_gated',
                'tier': 'medium_exact_first',
                'time': time.time() - t0,
            }
        if verbose:
            if exact_timed_out:
                print(
                    "Size-gated exact-first timed out; switching to hybrid "
                    f"(grid={grid_size}, timeout={exact_first_timeout}s)."
                )
            elif exact_err is not None:
                print(
                    "Size-gated exact-first errored; switching to hybrid "
                    f"(grid={grid_size}, err={exact_err})."
                )
            else:
                print(
                    "Size-gated exact-first found no solution quickly; "
                    "switching to hybrid."
                )

        res = hybrid_solve(
            pieces,
            grid_size=grid_size,
            model_name=model_name,
            beam_width=beam_width,
            timeout_nn=timeout_nn,
            timeout_dlx=timeout_dlx,
            device=device,
            verbose=verbose,
            **hybrid_kwargs,
        )
        res['controller'] = 'size_gated'
        res['tier'] = 'medium_exact_first'
        return res

    # Safety net: for medium grids (≤9) that reached the large tier because
    # exact_first_max_grid was set lower, try DLX before giving up.
    # DLX handles N=7 in ~6s, N=9 in ~30s for genuinely 3D pieces.
    if grid_size <= 9:
        dlx_budget = min(timeout_dlx, 90.0)
        fallback_sols, fb_timed_out, fb_err = _run_dlx_with_timeout(
            pieces, grid_size, dlx_budget,
        )
        if fallback_sols:
            return {
                'solution': fallback_sols[0],
                'method': 'dlx',
                'submethod': 'dlx_large_fallback',
                'controller': 'size_gated',
                'tier': 'large_dlx_fallback',
                'time': time.time() - t0,
            }

    # Large-cube tier: by default avoid exact fallback if no large model exists.
    if not model_name and not large_allow_dlx:
        return {
            'solution': None,
            'method': None,
            'submethod': (
                'block_planner_failed' if planner_diag is not None else 'no_model_large'
            ),
            'controller': 'size_gated',
            'tier': 'large_hybrid',
            'time': time.time() - t0,
            'planner_diag': planner_diag,
        }

    # Large-cube tier: currently routes to hybrid; planner wiring comes next.
    res = hybrid_solve(
        pieces,
        grid_size=grid_size,
        model_name=model_name,
        beam_width=beam_width,
        timeout_nn=timeout_nn,
        timeout_dlx=timeout_dlx,
        device=device,
        verbose=verbose,
        **hybrid_kwargs,
    )
    res['controller'] = 'size_gated'
    res['tier'] = 'large_hybrid'
    return res


def compare_solvers(pieces, grid_size=None, model_name="soma_3x3x3",
                    beam_width=None, device='cpu', verbose=True):
    """Run both NN and DLX solvers independently and compare results.

    Useful for benchmarking and evaluating the NN solver's quality.

    Returns:
        dict with results from both solvers
    """
    total_vol = sum(len(p) for p in pieces)
    if grid_size is None:
        grid_size = cube_root_int(total_vol)
        if grid_size is None:
            print(f"Volume {total_vol} is not a perfect cube.")
            return None

    results = {
        'grid_size': grid_size,
        'num_pieces': len(pieces),
        'total_volume': total_vol,
    }

    # DLX solver
    if verbose:
        print(f"DLX solver...")
    t0 = time.time()
    dlx_solutions = dlx_solve(pieces, grid_size=grid_size, find_all=False)
    dlx_time = time.time() - t0
    results['dlx_solved'] = len(dlx_solutions) > 0
    results['dlx_time'] = dlx_time
    results['dlx_solution'] = dlx_solutions[0] if dlx_solutions else None

    # NN solver
    try:
        model, _, metadata = load_model(model_name, device=device)
        max_pieces = model.in_channels - 1

        if verbose:
            print(f"NN solver (beam_width={beam_width})...")
        t0 = time.time()
        nn_solution = nn_solve(
            pieces, grid_size, model,
            max_pieces=max_pieces, beam_width=beam_width,
            timeout=30.0, device=device,
        )
        nn_time = time.time() - t0
        results['nn_solved'] = nn_solution is not None
        results['nn_time'] = nn_time
        results['nn_solution'] = nn_solution

    except FileNotFoundError:
        if verbose:
            print(f"No trained model '{model_name}' found.")
        results['nn_solved'] = False
        results['nn_time'] = 0.0
        results['nn_solution'] = None

    if verbose:
        print(f"\nResults:")
        print(f"  DLX: {'solved' if results['dlx_solved'] else 'no solution'} "
              f"in {results['dlx_time']:.4f}s")
        print(f"  NN:  {'solved' if results['nn_solved'] else 'no solution'} "
              f"in {results['nn_time']:.4f}s")
        if results['dlx_solved'] and results['nn_solved']:
            speedup = results['dlx_time'] / max(results['nn_time'], 1e-9)
            print(f"  Speedup: {speedup:.2f}x")

    return results


if __name__ == "__main__":
    from phase1.test_cases import SOMA_PIECES

    print("=" * 60)
    print("Hybrid Solver Test: Soma Cube")
    print("=" * 60)

    result = hybrid_solve(
        SOMA_PIECES, grid_size=3,
        model_name="soma_3x3x3_quick",
        beam_width=64,
    )

    if result['solution'] is not None:
        print(f"\nSolution found via {result['method']} in {result['time']:.4f}s:")
        for pidx, cells in sorted(result['solution'].items()):
            print(f"  Piece {pidx}: {sorted(cells)}")
    else:
        print("\nNo solution found.")
