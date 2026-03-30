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


class SearchState:
    """Represents a partial state during beam search."""

    __slots__ = ('grid', 'occupied', 'remaining_pieces', 'remaining_indices',
                 'placed', 'grid_size', 'score')

    def __init__(self, grid, occupied, remaining_pieces, remaining_indices,
                 placed, grid_size, score=0.0):
        self.grid = grid                          # np.array (N,N,N)
        self.occupied = occupied                    # set of (x,y,z) tuples
        self.remaining_pieces = remaining_pieces    # list of piece coord lists
        self.remaining_indices = remaining_indices  # original piece indices
        self.placed = placed                        # dict: piece_idx -> frozenset
        self.grid_size = grid_size
        self.score = score

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
        )


def nn_solve(pieces, grid_size, model, max_pieces=10, beam_width=32,
             timeout=30.0, device='cpu', return_search_trace=False,
             max_candidates_per_state=50):
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

    Returns:
        If return_search_trace=False:
            dict mapping piece_idx -> frozenset of cells, or None if no solution
        If return_search_trace=True:
            (solution_dict_or_None, search_trace)
            Always returns the tuple — caller checks solution for None.
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

    for depth in range(num_pieces):
        if time.time() - t0 > timeout:
            break

        candidates = []
        beam_policy_logits = _score_beam_policies(beam, model, max_pieces, device)

        for state_idx, state in enumerate(beam):
            if time.time() - t0 > timeout:
                break

            # Pick the MRV piece for THIS state (fewest valid placements)
            best_local_idx = _pick_mrv_piece(state, piece_placements)
            if best_local_idx is None:
                # Dead end — record this state as a negative if tracing
                if return_search_trace:
                    search_trace['pruned'].append(
                        _make_trace_entry(state, grid_size, max_pieces)
                    )
                continue

            orig_idx = state.remaining_indices[best_local_idx]
            placements = piece_placements[orig_idx]

            # Filter to non-overlapping placements
            valid_placements = [
                p for p in placements if not (p & state.occupied)
            ]

            # Cap candidates per state to avoid explosion.
            # Rank placements with the policy head so expansion is model-guided
            # and deterministic (no random sampling).
            if len(valid_placements) > max_candidates_per_state:
                valid_placements = _select_top_policy_placements(
                    state=state,
                    placements=valid_placements,
                    top_k=max_candidates_per_state,
                    policy_logits=beam_policy_logits[state_idx],
                )

            for placement in valid_placements:
                # Create new state with this piece placed
                new_state = state.copy()
                new_state.placed[orig_idx] = placement
                new_state.occupied |= placement
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
                    if return_search_trace:
                        return solution, search_trace
                    return solution
                # Check if this state creates isolated pocketshec
                if has_isolated_pockets(new_state.occupied, grid_size, new_state.remaining_pieces):
                    continue  # skip this state entirely

                candidates.append(new_state)

        if not candidates:
            break

        # Score all candidates with the value network
        _score_candidates(candidates, model, max_pieces, device)

        # Keep top beam_width
        candidates.sort(key=lambda s: s.score, reverse=True)
        beam = candidates[:beam_width]
        pruned = candidates[beam_width:]

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
    if return_search_trace:
        return None, search_trace
    return None


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
    best_idx = None
    best_count = float('inf')

    for local_idx in range(len(state.remaining_indices)):
        orig_idx = state.remaining_indices[local_idx]
        placements = piece_placements[orig_idx]
        count = sum(1 for p in placements if not (p & state.occupied))
        if count == 0:
            return None  # dead end — this piece can't be placed
        if count < best_count:
            best_count = count
            best_idx = local_idx

    return best_idx


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


def _select_top_policy_placements(state, placements, top_k, policy_logits):
    """Select top-k placements using policy logits from the current state.

    The current policy head emits N^3 logits over grid cells. We map a
    candidate placement to a scalar by averaging logits over its occupied cells.
    """
    if top_k <= 0 or not placements:
        return []
    logits = policy_logits

    n = state.grid_size
    n2 = n * n

    scored = []
    for placement in placements:
        flat_indices = [x * n2 + y * n + z for x, y, z in placement]
        score = float(np.mean(logits[flat_indices]))
        # Deterministic tie-breaker by sorted placement coordinates.
        tie_break = tuple(sorted(placement))
        scored.append((score, tie_break, placement))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:top_k]]


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


# ── Convenience Functions ─────────────────────────────────────────────────────

def solve_with_nn(pieces, grid_size=None, model_name="soma_3x3x3",
                  beam_width=32, timeout=30.0, device='cpu'):
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

    return nn_solve(
        pieces, grid_size, model,
        max_pieces=max_pieces, beam_width=beam_width,
        timeout=timeout, device=device,
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
def has_isolated_pockets(occupied, grid_size, remaining_pieces):
    """Check if empty space can be decomposed into fillable regions."""
    empty_cells = set()
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                if (x,y,z) not in occupied:
                    empty_cells.add((x,y,z))
    
    # Find connected components of empty space
    components = get_connected_components(empty_cells)
    
    remaining_sizes = [len(p) for p in remaining_pieces]
    
    # Each component's size must be achievable by some subset of remaining pieces
    for component in components:
        size = len(component)
        if size < min(remaining_sizes):  # too small for any piece
            return True  # isolated pocket — dead end, prune this branch
    
    return False

def get_connected_components(empty_cells):
    """Find connected components of empty cells using BFS."""
    unvisited = set(empty_cells)
    components = []
    neighbors = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    
    while unvisited:
        # Start a new component
        start = next(iter(unvisited))
        component = set()
        queue = [start]
        
        while queue:
            cell = queue.pop()
            if cell not in unvisited:
                continue
            unvisited.remove(cell)
            component.add(cell)
            x, y, z = cell
            for dx, dy, dz in neighbors:
                nb = (x+dx, y+dy, z+dz)
                if nb in unvisited:
                    queue.append(nb)
        
        components.append(component)
    
    return components