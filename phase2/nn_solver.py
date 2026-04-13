"""
NN-guided beam search solver for polycube packing.

Uses the trained CuboidNet to score partial states and guide search:
- Value head scores states by P(solvable)
- Policy head suggests which placements to try first
- Beam search keeps top-K most promising states at each depth
"""

import sys
import os
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1.polycube import get_all_placements
from phase2.data_generator import encode_state, encode_grid, encode_placement
from phase2.search_profiles import resolve_runtime_search_settings


class SearchState:
    """Represents a partial state during beam search."""

    __slots__ = ('grid', 'occupied', 'remaining_pieces', 'remaining_indices',
                 'placed', 'grid_size', 'score', 'last_piece_idx')

    def __init__(self, grid, occupied, remaining_pieces, remaining_indices,
                 placed, grid_size, score=0.0, last_piece_idx=None):
        self.grid = grid                          # np.array (N,N,N)
        self.occupied = occupied                    # set of (x,y,z) tuples
        self.remaining_pieces = remaining_pieces    # list of piece coord lists
        self.remaining_indices = remaining_indices  # original piece indices
        self.placed = placed                        # dict: piece_idx -> frozenset
        self.grid_size = grid_size
        self.score = score
        self.last_piece_idx = last_piece_idx

    def is_solved(self):
        return len(self.remaining_pieces) == 0

    def copy(self):
        return SearchState(
            grid=self.grid.copy(),
            occupied=set(self.occupied),
            remaining_pieces=list(self.remaining_pieces),
            remaining_indices=list(self.remaining_indices),
            placed=dict(self.placed),
            grid_size=self.grid_size,
            score=self.score,
            last_piece_idx=self.last_piece_idx,
        )


def _beam_diversity_signature(state, metric='slice_profile'):
    """Coarse structural signature for diversity-preserving beam selection."""
    if metric not in ('slice_profile', 'piece_slice_profile'):
        raise ValueError(f"Unsupported beam diversity metric '{metric}'")

    occupied = state.grid > 0.5
    x_counts = occupied.sum(axis=(1, 2), dtype=np.int16)
    y_counts = occupied.sum(axis=(0, 2), dtype=np.int16)
    z_counts = occupied.sum(axis=(0, 1), dtype=np.int16)

    mid = max(1, state.grid_size // 2)
    coarse = []
    for xs in ((0, mid), (mid, state.grid_size)):
        for ys in ((0, mid), (mid, state.grid_size)):
            for zs in ((0, mid), (mid, state.grid_size)):
                coarse.append(
                    int(occupied[xs[0]:xs[1], ys[0]:ys[1], zs[0]:zs[1]].sum())
                )

    base = tuple(int(v) for v in np.concatenate((x_counts, y_counts, z_counts))) + tuple(coarse)
    if metric == 'piece_slice_profile':
        piece_sig = -1 if state.last_piece_idx is None else int(state.last_piece_idx)
        return (piece_sig,) + base
    return base


def _select_beam_states(
    candidates,
    beam_width,
    *,
    diversity_slots=0,
    diversity_metric='slice_profile',
):
    """Keep top-scoring states while reserving a few slots for distinct frontiers."""
    ordered = sorted(candidates, key=lambda s: s.score, reverse=True)
    if beam_width <= 0:
        return [], ordered, 0, 0
    if len(ordered) <= beam_width:
        unique = len({_beam_diversity_signature(s, diversity_metric) for s in ordered})
        return ordered, [], 0, unique

    diversity_slots = int(max(0, diversity_slots or 0))
    diversity_slots = min(diversity_slots, max(0, beam_width - 1))
    if diversity_slots == 0:
        beam = ordered[:beam_width]
        pruned = ordered[beam_width:]
        unique = len({_beam_diversity_signature(s, diversity_metric) for s in beam})
        return beam, pruned, 0, unique

    primary_count = max(1, beam_width - diversity_slots)
    beam = list(ordered[:primary_count])
    selected_ids = {id(s) for s in beam}
    seen_signatures = {_beam_diversity_signature(s, diversity_metric) for s in beam}
    diverse_kept = 0

    for state in ordered[primary_count:]:
        if len(beam) >= beam_width:
            break
        sig = _beam_diversity_signature(state, diversity_metric)
        if sig in seen_signatures:
            continue
        beam.append(state)
        selected_ids.add(id(state))
        seen_signatures.add(sig)
        diverse_kept += 1

    for state in ordered[primary_count:]:
        if len(beam) >= beam_width:
            break
        if id(state) in selected_ids:
            continue
        beam.append(state)
        selected_ids.add(id(state))

    beam.sort(key=lambda s: s.score, reverse=True)
    pruned = [state for state in ordered if id(state) not in selected_ids]
    return beam, pruned, diverse_kept, len(seen_signatures)


def nn_solve(pieces, grid_size, model, max_pieces=10, beam_width=32,
             timeout=30.0, device='cpu', return_search_trace=False,
             max_candidates_per_state=50, enable_pocket_pruning=True,
             return_diagnostics=False, placement_ranker='policy',
             max_children_per_layer=None, frontier_out=None,
             pruned_frontier_out=None,
             beam_diversity_slots=0, beam_diversity_metric='slice_profile',
             piece_branching_width=1, piece_branching_slack=0):
    """Solve polycube packing using NN-guided beam search.

    At each depth:
    1. For each state in beam, pick the MRV piece (fewest valid placements)
    2. Enumerate that piece's valid placements, creating child states
    3. Score all children with the value network
    4. Keep top beam_width states (highest P(solvable))
    5. If any state has all pieces placed -> solution found

    Args:
        pieces: list of pieces (each a list of (x,y,z) tuples)
        grid_size: side length of target cube
        model: trained CuboidNet
        max_pieces: max piece channels for encoding
        beam_width: number of states to keep per depth level
        timeout: max seconds before giving up
        device: 'cpu' or 'cuda'
        return_search_trace: if True, also return search trace for ADI.
            The trace is a dict with 'kept' (states kept in beam) and
            'pruned' (states scored but dropped from beam).
        max_candidates_per_state: max child states to generate per beam state
        enable_pocket_pruning: if True, prune states that create
            disconnected empty-space pockets incompatible with remaining pieces
        return_diagnostics: if True, include search diagnostics
        placement_ranker: how to rank placements when truncating candidates.
            'policy' (default): use policy logits
            'contact': use geometric contact heuristic
            'hybrid': policy score with contact tie-break
        max_children_per_layer: optional hard cap on total generated children
            per search depth (controls NN scoring cost)
        frontier_out: optional list; when provided and solve fails, receives
            serialized frontier states from the final beam.
        pruned_frontier_out: optional list; when provided and solve fails,
            receives the score-ordered pruned states from the final beam layer.
        beam_diversity_slots: number of beam slots reserved for structurally
            distinct frontier states after score ranking.
        beam_diversity_metric: coarse structural signature used for diversity.
        piece_branching_width: number of near-MRV pieces to expand per beam
            state.
        piece_branching_slack: allow alternate pieces whose valid-placement
            count is within this slack of the MRV minimum.

    Returns:
        If return_search_trace=False and return_diagnostics=False:
            dict mapping piece_idx -> frozenset of cells, or None if no solution
        If return_search_trace=True and return_diagnostics=False:
            (solution_dict_or_None, search_trace)
            Always returns the tuple — caller checks solution for None.
        If return_search_trace=False and return_diagnostics=True:
            (solution_dict_or_None, diagnostics)
        If return_search_trace=True and return_diagnostics=True:
            (solution_dict_or_None, search_trace, diagnostics)
    """
    model.eval()
    model = model.to(device)
    t0 = time.time()

    num_pieces = len(pieces)

    # Precompute all valid placements for each piece
    piece_placements = []
    for piece in pieces:
        placements = get_all_placements(piece, grid_size)
        piece_placements.append(placements)

    # Initialize beam with empty grid
    initial_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    initial_state = SearchState(
        grid=initial_grid,
        occupied=set(),
        remaining_pieces=list(pieces),
        remaining_indices=list(range(num_pieces)),
        placed={},
        grid_size=grid_size,
        score=1.0,
    )

    beam = [initial_state]
    search_trace = {'kept': [], 'pruned': []}
    diagnostics = {
        'timed_out': False,
        'failure_reason': None,
        'elapsed_sec': 0.0,
        'depth_reached': 0,
        'solved_depth': None,
        'expanded_states': 0,
        'dead_end_states': 0,
        'generated_children': 0,
        'pocket_pruned_children': 0,
        'value_scored_children': 0,
        'truncated_states': 0,
        'placements_considered_total': 0,
        'placements_dropped_total': 0,
        'beam_max_size_seen': 1,
        'beam_nodes_kept_total': 1,
        'beam_diverse_kept_total': 0,
        'beam_unique_signatures_max': 1,
        'beam_unique_signatures_last': 1,
        'layer_budget_truncated_states': 0,
        'layer_budget_dropped_total': 0,
        'piece_branching_states': 0,
        'piece_branching_alternates': 0,
    }
    final_pruned = []

    for depth in range(num_pieces):
        if time.time() - t0 > timeout:
            diagnostics['timed_out'] = True
            diagnostics['failure_reason'] = 'timeout'
            break
        diagnostics['depth_reached'] = max(diagnostics['depth_reached'], depth)

        candidates = []
        beam_policy_logits = None
        if placement_ranker in ('policy', 'hybrid'):
            beam_policy_logits = _score_beam_policies(beam, model, max_pieces, device)

        for state_idx, state in enumerate(beam):
            diagnostics['expanded_states'] += 1
            if time.time() - t0 > timeout:
                diagnostics['timed_out'] = True
                diagnostics['failure_reason'] = 'timeout'
                break

            piece_options = _pick_piece_options(
                state,
                piece_placements,
                branching_width=piece_branching_width,
                branching_slack=piece_branching_slack,
            )
            if piece_options is None:
                diagnostics['dead_end_states'] += 1
                # Dead end — record this state as a negative if tracing
                if return_search_trace:
                    search_trace['pruned'].append(
                        _make_trace_entry(state, grid_size, max_pieces)
                    )
                continue

            diagnostics['piece_branching_states'] += 1
            diagnostics['piece_branching_alternates'] += max(0, len(piece_options) - 1)
            policy_logits = None
            if beam_policy_logits is not None:
                policy_logits = beam_policy_logits[state_idx]

            quotas = _allocate_piece_branch_quotas(
                len(piece_options),
                max_candidates_per_state,
            )
            ranked_piece_options = []
            for piece_option, option_quota in zip(piece_options, quotas):
                valid_placements = list(piece_option['valid_placements'])
                diagnostics['placements_considered_total'] += len(valid_placements)
                if option_quota is not None and len(valid_placements) > option_quota:
                    diagnostics['truncated_states'] += 1
                    diagnostics['placements_dropped_total'] += (
                        len(valid_placements) - option_quota
                    )
                ranked_piece_options.append(
                    {
                        'local_idx': piece_option['local_idx'],
                        'orig_idx': piece_option['orig_idx'],
                        'placements': _select_top_placements(
                            state=state,
                            placements=valid_placements,
                            top_k=min(len(valid_placements), option_quota or len(valid_placements)),
                            policy_logits=policy_logits,
                            ranker=placement_ranker,
                        ),
                    }
                )

            # Optional global cap per depth to prevent combinatorial blow-up.
            if max_children_per_layer is not None and max_children_per_layer > 0:
                remaining_budget = max_children_per_layer - len(candidates)
                if remaining_budget <= 0:
                    continue
                remaining_states = max(1, len(beam) - state_idx)
                layer_cap = max(1, remaining_budget // remaining_states)
                total_before_layer_cap = sum(len(item['placements']) for item in ranked_piece_options)
                if total_before_layer_cap > layer_cap:
                    diagnostics['layer_budget_truncated_states'] += 1
                    diagnostics['layer_budget_dropped_total'] += (
                        total_before_layer_cap - layer_cap
                    )
                    ranked_piece_options = _apply_layer_budget_across_pieces(
                        ranked_piece_options,
                        layer_cap,
                    )

            for piece_option in ranked_piece_options:
                best_local_idx = piece_option['local_idx']
                orig_idx = piece_option['orig_idx']
                for placement in piece_option['placements']:
                    # Create new state with this piece placed
                    new_state = state.copy()
                    new_state.placed[orig_idx] = placement
                    new_state.occupied |= placement
                    new_state.last_piece_idx = orig_idx
                    for x, y, z in placement:
                        new_state.grid[x, y, z] = 1.0

                    # Remove the placed piece from remaining
                    new_state.remaining_pieces = (
                        state.remaining_pieces[:best_local_idx] +
                        state.remaining_pieces[best_local_idx + 1:]
                    )
                    new_state.remaining_indices = (
                        state.remaining_indices[:best_local_idx] +
                        state.remaining_indices[best_local_idx + 1:]
                    )

                    # Check if solved
                    if new_state.is_solved():
                        solution = new_state.placed
                        diagnostics['solved_depth'] = depth + 1
                        diagnostics['failure_reason'] = None
                        diagnostics['elapsed_sec'] = time.time() - t0
                        if return_search_trace and return_diagnostics:
                            return solution, search_trace, diagnostics
                        if return_search_trace:
                            return solution, search_trace
                        if return_diagnostics:
                            return solution, diagnostics
                        return solution
                    if (
                        enable_pocket_pruning
                        and grid_size >= 4
                        and new_state.remaining_pieces
                        and has_isolated_pockets(
                            new_state.occupied, grid_size, new_state.remaining_pieces
                        )
                    ):
                        diagnostics['pocket_pruned_children'] += 1
                        continue

                    candidates.append(new_state)
                    diagnostics['generated_children'] += 1

        if not candidates:
            if diagnostics['failure_reason'] is None:
                diagnostics['failure_reason'] = 'no_candidates'
            break

        # Score all candidates with the value network
        _score_candidates(candidates, model, max_pieces, device)
        diagnostics['value_scored_children'] += len(candidates)

        # Keep a score-ordered beam while preserving some structural diversity.
        beam, pruned, diverse_kept, unique_signatures = _select_beam_states(
            candidates,
            beam_width,
            diversity_slots=beam_diversity_slots,
            diversity_metric=beam_diversity_metric,
        )
        final_pruned = pruned
        diagnostics['beam_nodes_kept_total'] += len(beam)
        diagnostics['beam_diverse_kept_total'] += diverse_kept
        diagnostics['beam_unique_signatures_last'] = unique_signatures
        diagnostics['beam_unique_signatures_max'] = max(
            diagnostics['beam_unique_signatures_max'],
            unique_signatures,
        )
        diagnostics['beam_max_size_seen'] = max(
            diagnostics['beam_max_size_seen'], len(beam)
        )

        # Record search trace for ADI
        if return_search_trace:
            for state in beam:
                search_trace['kept'].append(
                    _make_trace_entry(state, grid_size, max_pieces)
                )
            # Sample some pruned states as negative examples (not all — too many)
            prune_sample = pruned[:beam_width] if len(pruned) > beam_width else pruned
            for state in prune_sample:
                search_trace['pruned'].append(
                    _make_trace_entry(state, grid_size, max_pieces)
                )

    # No solution found
    if diagnostics['failure_reason'] is None:
        diagnostics['failure_reason'] = 'search_exhausted'
    diagnostics['elapsed_sec'] = time.time() - t0
    if frontier_out is not None:
        frontier_out.clear()
        frontier_out.extend(_serialize_state(s) for s in beam)
    if pruned_frontier_out is not None:
        pruned_frontier_out.clear()
        pruned_frontier_out.extend(_serialize_state(s) for s in final_pruned)

    if return_search_trace and return_diagnostics:
        return None, search_trace, diagnostics
    if return_search_trace:
        return None, search_trace
    if return_diagnostics:
        return None, diagnostics
    return None


def _serialize_state(state):
    """Serialize SearchState into a plain dict for retry workflows."""
    return {
        'grid': state.grid.copy(),
        'occupied': set(state.occupied),
        'remaining_pieces': [list(p) for p in state.remaining_pieces],
        'remaining_indices': list(state.remaining_indices),
        'placed': {k: frozenset(v) for k, v in state.placed.items()},
        'grid_size': state.grid_size,
        'score': float(state.score),
        'last_piece_idx': state.last_piece_idx,
    }


def _deserialize_state(state_dict):
    """Reconstruct SearchState from serialized dict."""
    return SearchState(
        grid=state_dict['grid'].copy(),
        occupied=set(state_dict['occupied']),
        remaining_pieces=[list(p) for p in state_dict['remaining_pieces']],
        remaining_indices=list(state_dict['remaining_indices']),
        placed={int(k): frozenset(v) for k, v in state_dict['placed'].items()},
        grid_size=int(state_dict['grid_size']),
        score=float(state_dict.get('score', 0.0)),
        last_piece_idx=state_dict.get('last_piece_idx'),
    )


def dfs_solve_from_frontier(
    pieces,
    grid_size,
    model,
    frontier_states,
    max_pieces=10,
    timeout=20.0,
    max_frontier_states=16,
    branch_limit=12,
    max_nodes=15000,
    device='cpu',
    placement_ranker='contact',
    enable_pocket_pruning=True,
    use_transposition=True,
):
    """Bounded DFS retry over frontier states from failed beam search.

    Returns:
        (solution_or_none, diagnostics_dict)
    """
    model.eval()
    model = model.to(device)
    t0 = time.time()

    diagnostics = {
        'timed_out': False,
        'elapsed_sec': 0.0,
        'roots_considered': 0,
        'nodes_expanded': 0,
        'dead_ends': 0,
        'generated_children': 0,
        'pocket_pruned_children': 0,
        'value_scored_children': 0,
        'truncated_nodes': 0,
        'transposition_hits': 0,
        'visited_states': 0,
        'max_stack_size': 0,
    }

    if not frontier_states:
        diagnostics['elapsed_sec'] = time.time() - t0
        return None, diagnostics

    piece_placements = [get_all_placements(piece, grid_size) for piece in pieces]
    roots = sorted(
        frontier_states,
        key=lambda s: float(s.get('score', 0.0)),
        reverse=True,
    )[:max_frontier_states]

    visited = set()

    for root_dict in roots:
        if time.time() - t0 > timeout or diagnostics['nodes_expanded'] >= max_nodes:
            diagnostics['timed_out'] = True
            break

        root = _deserialize_state(root_dict)
        diagnostics['roots_considered'] += 1
        stack = [root]

        while stack:
            diagnostics['max_stack_size'] = max(diagnostics['max_stack_size'], len(stack))
            if time.time() - t0 > timeout:
                diagnostics['timed_out'] = True
                stack.clear()
                break
            if diagnostics['nodes_expanded'] >= max_nodes:
                stack.clear()
                break

            state = stack.pop()
            diagnostics['nodes_expanded'] += 1

            if state.is_solved():
                diagnostics['elapsed_sec'] = time.time() - t0
                return state.placed, diagnostics

            if use_transposition:
                key = (frozenset(state.occupied), tuple(sorted(state.remaining_indices)))
                if key in visited:
                    diagnostics['transposition_hits'] += 1
                    continue
                visited.add(key)
                diagnostics['visited_states'] = len(visited)

            best_local_idx = _pick_mrv_piece(state, piece_placements)
            if best_local_idx is None:
                diagnostics['dead_ends'] += 1
                continue

            orig_idx = state.remaining_indices[best_local_idx]
            placements = piece_placements[orig_idx]
            valid = [p for p in placements if not (p & state.occupied)]
            if not valid:
                diagnostics['dead_ends'] += 1
                continue

            if len(valid) > branch_limit:
                diagnostics['truncated_nodes'] += 1
                policy_logits = None
                if placement_ranker in ('policy', 'hybrid'):
                    policy_logits = _score_beam_policies(
                        [state], model, max_pieces, device
                    )[0]
                valid = _select_top_placements(
                    state=state,
                    placements=valid,
                    top_k=branch_limit,
                    policy_logits=policy_logits,
                    ranker=placement_ranker,
                )

            children = []
            for placement in valid:
                child = state.copy()
                child.placed[orig_idx] = placement
                child.occupied |= placement
                for x, y, z in placement:
                    child.grid[x, y, z] = 1.0
                child.remaining_pieces = (
                    state.remaining_pieces[:best_local_idx] +
                    state.remaining_pieces[best_local_idx + 1:]
                )
                child.remaining_indices = (
                    state.remaining_indices[:best_local_idx] +
                    state.remaining_indices[best_local_idx + 1:]
                )

                if child.is_solved():
                    diagnostics['elapsed_sec'] = time.time() - t0
                    return child.placed, diagnostics

                if (
                    enable_pocket_pruning
                    and grid_size >= 4
                    and child.remaining_pieces
                    and has_isolated_pockets(
                        child.occupied, grid_size, child.remaining_pieces
                    )
                ):
                    diagnostics['pocket_pruned_children'] += 1
                    continue

                children.append(child)
                diagnostics['generated_children'] += 1

            if not children:
                diagnostics['dead_ends'] += 1
                continue

            _score_candidates(children, model, max_pieces, device)
            diagnostics['value_scored_children'] += len(children)
            # Push lowest score first so highest score gets popped next.
            children.sort(key=lambda s: s.score)
            stack.extend(children)

    diagnostics['elapsed_sec'] = time.time() - t0
    return None, diagnostics


def complete_solve_ordered(
    pieces,
    grid_size,
    model,
    max_pieces=10,
    timeout=60.0,
    max_nodes=200000,
    device='cpu',
    placement_ranker='contact',
    enable_pocket_pruning=True,
    use_transposition=True,
):
    """Deterministic complete backtracking search with ordered expansion.

    This is exhaustive up to timeout/node budget limits:
    - MRV piece selection (same as beam/DFS helpers)
    - deterministic placement ordering via policy/contact/hybrid ranking
    - optional transposition table and pocket pruning

    Returns:
        (solution_or_none, diagnostics_dict)
    """
    model.eval()
    model = model.to(device)
    t0 = time.time()

    piece_placements = [get_all_placements(piece, grid_size) for piece in pieces]
    num_pieces = len(pieces)
    initial_state = SearchState(
        grid=np.zeros((grid_size, grid_size, grid_size), dtype=np.float32),
        occupied=set(),
        remaining_pieces=list(pieces),
        remaining_indices=list(range(num_pieces)),
        placed={},
        grid_size=grid_size,
        score=1.0,
    )

    diagnostics = {
        'timed_out': False,
        'failure_reason': None,
        'elapsed_sec': 0.0,
        'nodes_expanded': 0,
        'dead_ends': 0,
        'generated_children': 0,
        'pocket_pruned_children': 0,
        'placements_considered_total': 0,
        'depth_reached': 0,
        'solved_depth': None,
        'transposition_hits': 0,
        'visited_states': 0,
        'max_stack_size': 1,
    }

    stack = [initial_state]
    visited = set()

    while stack:
        diagnostics['max_stack_size'] = max(diagnostics['max_stack_size'], len(stack))
        if time.time() - t0 > timeout:
            diagnostics['timed_out'] = True
            diagnostics['failure_reason'] = 'timeout'
            break
        if max_nodes is not None and max_nodes > 0 and diagnostics['nodes_expanded'] >= max_nodes:
            diagnostics['failure_reason'] = 'max_nodes'
            break

        state = stack.pop()
        diagnostics['nodes_expanded'] += 1
        depth = len(state.placed)
        diagnostics['depth_reached'] = max(diagnostics['depth_reached'], depth)

        if state.is_solved():
            diagnostics['solved_depth'] = depth
            diagnostics['failure_reason'] = None
            diagnostics['elapsed_sec'] = time.time() - t0
            return state.placed, diagnostics

        if use_transposition:
            key = (frozenset(state.occupied), tuple(sorted(state.remaining_indices)))
            if key in visited:
                diagnostics['transposition_hits'] += 1
                continue
            visited.add(key)
            diagnostics['visited_states'] = len(visited)

        best_local_idx = _pick_mrv_piece(state, piece_placements)
        if best_local_idx is None:
            diagnostics['dead_ends'] += 1
            continue

        orig_idx = state.remaining_indices[best_local_idx]
        placements = piece_placements[orig_idx]
        valid_placements = [p for p in placements if not (p & state.occupied)]
        diagnostics['placements_considered_total'] += len(valid_placements)
        if not valid_placements:
            diagnostics['dead_ends'] += 1
            continue

        policy_logits = None
        if placement_ranker in ('policy', 'hybrid'):
            policy_logits = _score_beam_policies([state], model, max_pieces, device)[0]
        ordered_placements = _select_top_placements(
            state=state,
            placements=valid_placements,
            top_k=len(valid_placements),
            policy_logits=policy_logits,
            ranker=placement_ranker,
        )

        children = []
        for placement in ordered_placements:
            child = state.copy()
            child.placed[orig_idx] = placement
            child.occupied |= placement
            for x, y, z in placement:
                child.grid[x, y, z] = 1.0
            child.remaining_pieces = (
                state.remaining_pieces[:best_local_idx] +
                state.remaining_pieces[best_local_idx + 1:]
            )
            child.remaining_indices = (
                state.remaining_indices[:best_local_idx] +
                state.remaining_indices[best_local_idx + 1:]
            )

            if child.is_solved():
                diagnostics['solved_depth'] = depth + 1
                diagnostics['failure_reason'] = None
                diagnostics['elapsed_sec'] = time.time() - t0
                return child.placed, diagnostics

            if (
                enable_pocket_pruning
                and grid_size >= 4
                and child.remaining_pieces
                and has_isolated_pockets(
                    child.occupied, grid_size, child.remaining_pieces
                )
            ):
                diagnostics['pocket_pruned_children'] += 1
                continue

            children.append(child)
            diagnostics['generated_children'] += 1

        if not children:
            diagnostics['dead_ends'] += 1
            continue

        # LIFO stack: push reverse order so the highest-ranked child is explored first.
        for child in reversed(children):
            stack.append(child)

    if diagnostics['failure_reason'] is None:
        diagnostics['failure_reason'] = 'search_exhausted'
    diagnostics['elapsed_sec'] = time.time() - t0
    return None, diagnostics


def complete_solve_from_frontier(
    pieces,
    grid_size,
    model,
    frontier_states,
    max_pieces=10,
    timeout=60.0,
    max_nodes=200000,
    max_frontier_states=8,
    device='cpu',
    placement_ranker='contact',
    enable_pocket_pruning=True,
    use_transposition=True,
):
    """Deterministic complete search seeded from frontier states.

    This answers a specific diagnostic question:
    if beam search fails, is the saved frontier still recoverable under a
    stronger local search, or has the policy already driven search into a bad
    region?

    Returns:
        (solution_or_none, diagnostics_dict)
    """
    model.eval()
    model = model.to(device)
    t0 = time.time()

    diagnostics = {
        'timed_out': False,
        'failure_reason': None,
        'elapsed_sec': 0.0,
        'roots_considered': 0,
        'nodes_expanded': 0,
        'dead_ends': 0,
        'generated_children': 0,
        'pocket_pruned_children': 0,
        'placements_considered_total': 0,
        'depth_reached': 0,
        'solved_depth': None,
        'transposition_hits': 0,
        'visited_states': 0,
        'max_stack_size': 0,
    }

    if not frontier_states:
        diagnostics['failure_reason'] = 'no_frontier'
        diagnostics['elapsed_sec'] = time.time() - t0
        return None, diagnostics

    piece_placements = [get_all_placements(piece, grid_size) for piece in pieces]
    roots = sorted(
        frontier_states,
        key=lambda s: float(s.get('score', 0.0)),
        reverse=True,
    )[:max_frontier_states]
    diagnostics['roots_considered'] = len(roots)

    stack = [_deserialize_state(root) for root in reversed(roots)]
    diagnostics['max_stack_size'] = len(stack)
    visited = set()

    while stack:
        diagnostics['max_stack_size'] = max(diagnostics['max_stack_size'], len(stack))
        if time.time() - t0 > timeout:
            diagnostics['timed_out'] = True
            diagnostics['failure_reason'] = 'timeout'
            break
        if max_nodes is not None and max_nodes > 0 and diagnostics['nodes_expanded'] >= max_nodes:
            diagnostics['failure_reason'] = 'max_nodes'
            break

        state = stack.pop()
        diagnostics['nodes_expanded'] += 1
        depth = len(state.placed)
        diagnostics['depth_reached'] = max(diagnostics['depth_reached'], depth)

        if state.is_solved():
            diagnostics['solved_depth'] = depth
            diagnostics['failure_reason'] = None
            diagnostics['elapsed_sec'] = time.time() - t0
            return state.placed, diagnostics

        if use_transposition:
            key = (frozenset(state.occupied), tuple(sorted(state.remaining_indices)))
            if key in visited:
                diagnostics['transposition_hits'] += 1
                continue
            visited.add(key)
            diagnostics['visited_states'] = len(visited)

        best_local_idx = _pick_mrv_piece(state, piece_placements)
        if best_local_idx is None:
            diagnostics['dead_ends'] += 1
            continue

        orig_idx = state.remaining_indices[best_local_idx]
        placements = piece_placements[orig_idx]
        valid_placements = [p for p in placements if not (p & state.occupied)]
        diagnostics['placements_considered_total'] += len(valid_placements)
        if not valid_placements:
            diagnostics['dead_ends'] += 1
            continue

        policy_logits = None
        if placement_ranker in ('policy', 'hybrid'):
            policy_logits = _score_beam_policies([state], model, max_pieces, device)[0]
        ordered_placements = _select_top_placements(
            state=state,
            placements=valid_placements,
            top_k=len(valid_placements),
            policy_logits=policy_logits,
            ranker=placement_ranker,
        )

        children = []
        for placement in ordered_placements:
            child = state.copy()
            child.placed[orig_idx] = placement
            child.occupied |= placement
            for x, y, z in placement:
                child.grid[x, y, z] = 1.0
            child.remaining_pieces = (
                state.remaining_pieces[:best_local_idx] +
                state.remaining_pieces[best_local_idx + 1:]
            )
            child.remaining_indices = (
                state.remaining_indices[:best_local_idx] +
                state.remaining_indices[best_local_idx + 1:]
            )

            if child.is_solved():
                diagnostics['solved_depth'] = depth + 1
                diagnostics['failure_reason'] = None
                diagnostics['elapsed_sec'] = time.time() - t0
                return child.placed, diagnostics

            if (
                enable_pocket_pruning
                and grid_size >= 4
                and child.remaining_pieces
                and has_isolated_pockets(
                    child.occupied, grid_size, child.remaining_pieces
                )
            ):
                diagnostics['pocket_pruned_children'] += 1
                continue

            children.append(child)
            diagnostics['generated_children'] += 1

        if not children:
            diagnostics['dead_ends'] += 1
            continue

        for child in reversed(children):
            stack.append(child)

    if diagnostics['failure_reason'] is None:
        diagnostics['failure_reason'] = 'search_exhausted'
    diagnostics['elapsed_sec'] = time.time() - t0
    return None, diagnostics


def _make_trace_entry(state, grid_size, max_pieces):
    """Create a trace dict from a SearchState for ADI data collection."""
    return {
        'state': encode_state(
            state.grid, state.remaining_pieces, grid_size, max_pieces
        ),
        'grid': state.grid.copy(),
        'remaining_pieces': list(state.remaining_pieces),
        'value': len(state.remaining_pieces),
        'score': state.score,
        'grid_size': grid_size,
    }


def _pick_mrv_piece(state, piece_placements):
    """Pick the remaining piece with fewest valid placements (MRV heuristic).

    Returns the local index into state.remaining_pieces, or None if dead end.
    """
    options = _pick_piece_options(
        state,
        piece_placements,
        branching_width=1,
        branching_slack=0,
    )
    if not options:
        return None
    return options[0]['local_idx']


def _pick_piece_options(state, piece_placements, branching_width=1, branching_slack=0):
    """Return near-MRV piece options together with valid placements."""
    branching_width = max(1, int(branching_width or 1))
    branching_slack = max(0, int(branching_slack or 0))

    options = []
    best_count = float('inf')
    for local_idx in range(len(state.remaining_indices)):
        orig_idx = state.remaining_indices[local_idx]
        placements = piece_placements[orig_idx]
        valid_placements = [p for p in placements if not (p & state.occupied)]
        count = len(valid_placements)
        if count == 0:
            return None
        best_count = min(best_count, count)
        options.append(
            {
                'local_idx': local_idx,
                'orig_idx': orig_idx,
                'valid_count': count,
                'valid_placements': valid_placements,
                'piece_size': len(state.remaining_pieces[local_idx]),
            }
        )

    options.sort(
        key=lambda item: (
            item['valid_count'],
            -item['piece_size'],
            item['orig_idx'],
        )
    )

    selected = []
    cutoff = best_count + branching_slack
    for option in options:
        if selected and option['valid_count'] > cutoff and len(selected) >= 1:
            break
        selected.append(option)
        if len(selected) >= branching_width:
            break

    return selected or options[:1]


def _allocate_piece_branch_quotas(n_options, total_cap):
    """Allocate a per-piece placement budget across piece branches."""
    if n_options <= 0:
        return []
    if total_cap is None:
        return [None] * n_options

    total_cap = max(1, int(total_cap))
    n_options = min(n_options, total_cap)
    base = total_cap // n_options
    remainder = total_cap % n_options
    return [
        max(1, base + (1 if idx < remainder else 0))
        for idx in range(n_options)
    ]


def _apply_layer_budget_across_pieces(piece_options, layer_cap):
    """Interleave ranked placements across piece branches under a shared cap."""
    if layer_cap is None or layer_cap <= 0 or not piece_options:
        return piece_options

    queues = [list(option['placements']) for option in piece_options]
    kept = [[] for _ in piece_options]
    kept_total = 0
    while kept_total < layer_cap:
        progressed = False
        for idx, queue in enumerate(queues):
            if kept_total >= layer_cap:
                break
            if not queue:
                continue
            kept[idx].append(queue.pop(0))
            kept_total += 1
            progressed = True
        if not progressed:
            break

    out = []
    for option, kept_placements in zip(piece_options, kept):
        updated = dict(option)
        updated['placements'] = kept_placements
        out.append(updated)
    return out


def _score_beam_policies(beam, model, max_pieces, device):
    """Compute policy logits for every state in the beam in one batch."""
    if not beam:
        return []

    batch_size = len(beam)
    grid_size = beam[0].grid_size
    states = np.zeros(
        (batch_size, 1 + max_pieces, grid_size, grid_size, grid_size),
        dtype=np.float32
    )
    for i, state in enumerate(beam):
        states[i] = encode_state(
            state.grid, state.remaining_pieces, grid_size, max_pieces
        )

    with torch.no_grad():
        state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        _, policy_logits = model(state_tensor)

    return [policy_logits[i].detach().cpu().numpy() for i in range(batch_size)]


def _select_top_placements(state, placements, top_k, policy_logits, ranker='policy'):
    """Select top-k placements using policy and/or geometric heuristics."""
    if top_k <= 0 or not placements:
        return []

    n = state.grid_size
    n2 = n * n

    scored = []
    if ranker not in ('policy', 'contact', 'hybrid'):
        ranker = 'policy'

    for placement in placements:
        tie_break = tuple(sorted(placement))
        contact = _placement_contact_score(state, placement)

        if ranker == 'contact' or policy_logits is None:
            scored.append((contact, tie_break, placement))
            continue

        flat_indices = [x * n2 + y * n + z for x, y, z in placement]
        policy_score = float(np.mean(policy_logits[flat_indices]))

        if ranker == 'hybrid':
            scored.append((policy_score, contact, tie_break, placement))
        else:  # ranker == 'policy'
            scored.append((policy_score, tie_break, placement))

    if ranker == 'hybrid' and policy_logits is not None:
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [item[3] for item in scored[:top_k]]

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:top_k]]


def _placement_contact_score(state, placement):
    """Heuristic: prefer placements with higher face contact to walls/occupied."""
    occupied = state.occupied
    n = state.grid_size
    neighbors = (
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    )
    contact = 0
    for x, y, z in placement:
        for dx, dy, dz in neighbors:
            nx, ny, nz = x + dx, y + dy, z + dz
            if nx < 0 or ny < 0 or nz < 0 or nx >= n or ny >= n or nz >= n:
                contact += 1
            elif (nx, ny, nz) in occupied:
                contact += 1
    return float(contact) / float(max(1, len(placement)))


def _score_candidates(candidates, model, max_pieces, device):
    """Score a batch of candidate states using the value network.

    Updates each candidate's .score attribute in place.
    """
    if not candidates:
        return

    # Batch encode all candidates
    batch_size = len(candidates)
    grid_size = candidates[0].grid_size

    states = np.zeros(
        (batch_size, 1 + max_pieces, grid_size, grid_size, grid_size),
        dtype=np.float32
    )
    for i, cand in enumerate(candidates):
        states[i] = encode_state(
            cand.grid, cand.remaining_pieces, grid_size, max_pieces
        )

    # Run value network
    with torch.no_grad():
        state_tensor = torch.tensor(states, dtype=torch.float32, device=device)

        # Process in sub-batches if too large
        sub_batch = 256
        scores = []
        for start in range(0, batch_size, sub_batch):
            end = min(start + sub_batch, batch_size)
            value, _ = model(state_tensor[start:end])
            scores.append(value.squeeze(-1).cpu().numpy())

        all_scores = np.concatenate(scores)

    for i, cand in enumerate(candidates):
        cand.score = float(all_scores[i])


def has_isolated_pockets(occupied, grid_size, remaining_pieces):
    """Return True if empty-space components cannot be tiled by remaining sizes.

    Uses inexpensive necessary conditions:
    - component size >= min remaining piece size
    - component size divisible by gcd of remaining piece sizes
    - component size present in subset-sum reachable volumes
    """
    remaining_sizes = [len(p) for p in remaining_pieces]
    if not remaining_sizes:
        return False

    min_size = min(remaining_sizes)
    gcd_size = _gcd_list(remaining_sizes)
    reachable = _subset_sum_reachable(remaining_sizes)

    empty_cells = set()
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                if (x, y, z) not in occupied:
                    empty_cells.add((x, y, z))

    for comp_size in _connected_component_sizes(empty_cells):
        if comp_size < min_size:
            return True
        if comp_size % gcd_size != 0:
            return True
        if comp_size >= len(reachable) or not reachable[comp_size]:
            return True
    return False


def _connected_component_sizes(empty_cells):
    """Yield connected component sizes of empty cells (6-neighborhood)."""
    unvisited = set(empty_cells)
    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                 (0, -1, 0), (0, 0, 1), (0, 0, -1))

    while unvisited:
        start = next(iter(unvisited))
        stack = [start]
        size = 0
        while stack:
            cell = stack.pop()
            if cell not in unvisited:
                continue
            unvisited.remove(cell)
            size += 1
            x, y, z = cell
            for dx, dy, dz in neighbors:
                nb = (x + dx, y + dy, z + dz)
                if nb in unvisited:
                    stack.append(nb)
        yield size


def _subset_sum_reachable(sizes):
    """Return boolean array where idx is reachable subset volume."""
    total = sum(sizes)
    reachable = np.zeros(total + 1, dtype=bool)
    reachable[0] = True
    for s in sizes:
        # backward update for 0/1 subset sum
        reachable[s:] = reachable[s:] | reachable[:-s]
    return reachable


def _gcd_list(nums):
    from math import gcd
    g = nums[0]
    for n in nums[1:]:
        g = gcd(g, n)
    return g


# ── Convenience Functions ─────────────────────────────────────────────────────

def solve_with_nn(pieces, grid_size=None, model_name="soma_3x3x3",
                  beam_width=None, timeout=30.0, device='cpu',
                  max_candidates_per_state=None, placement_ranker=None,
                  enable_pocket_pruning=None, max_children_per_layer=None,
                  beam_diversity_slots=None, beam_diversity_metric=None,
                  piece_branching_width=None, piece_branching_slack=None):
    """High-level interface: load a trained model and solve.

    Args:
        pieces: list of pieces
        grid_size: target cube size (auto-detected if None)
        model_name: name of saved model to load
        beam_width: beam search width
        timeout: max seconds
        device: 'cpu' or 'cuda'

    Returns:
        solution dict or None
    """
    from phase2.train import load_model
    from phase1.solver import cube_root_int

    if grid_size is None:
        total = sum(len(p) for p in pieces)
        grid_size = cube_root_int(total)
        if grid_size is None:
            return None

    model, _, metadata = load_model(model_name, device=device)
    max_pieces = model.in_channels - 1

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
        default_beam_width=32,
        default_max_candidates_per_state=50,
    )

    return nn_solve(
        pieces, grid_size, model,
        max_pieces=max_pieces, beam_width=runtime["beam_width"],
        timeout=timeout, device=device,
        max_candidates_per_state=runtime["max_candidates_per_state"],
        enable_pocket_pruning=runtime["enable_pocket_pruning"],
        placement_ranker=runtime["placement_ranker"],
        max_children_per_layer=runtime["max_children_per_layer"],
        beam_diversity_slots=runtime["beam_diversity_slots"],
        beam_diversity_metric=runtime["beam_diversity_metric"],
        piece_branching_width=runtime["piece_branching_width"],
        piece_branching_slack=runtime["piece_branching_slack"],
    )


if __name__ == "__main__":
    # Quick test: try to solve Soma cube with a trained model
    from phase1.test_cases import SOMA_PIECES

    print("Testing NN solver on Soma cube...")
    print("(Requires a trained model — run train.py first)")

    try:
        solution = solve_with_nn(
            SOMA_PIECES, grid_size=3,
            model_name="soma_3x3x3_quick",
            beam_width=64, timeout=30.0,
        )

        if solution is not None:
            print("Solution found!")
            for pidx, cells in sorted(solution.items()):
                print(f"  Piece {pidx}: {sorted(cells)}")
        else:
            print("No solution found by NN solver.")
    except FileNotFoundError:
        print("No trained model found. Run train.py first.")
