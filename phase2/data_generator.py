"""
Training data generation for the neural polycube solver.

Generates solvable puzzle instances using DLX, then creates partial-state
training examples (positive = solvable, negative = unsolvable) for
supervised learning of value and policy networks.
"""

import sys
import os
import random
import time
from collections import deque
from copy import deepcopy

import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1.polycube import normalize, get_orientations, get_all_placements
from phase1.solver import solve, cube_root_int
from phase1.dlx_solver import DLX


# ── Polycube Enumeration ──────────────────────────────────────────────────────

def _neighbors_3d(cell):
    """Return the 6 face-adjacent neighbors of a 3D cell."""
    x, y, z = cell
    return [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y, z + 1), (x, y, z - 1),
    ]


def enumerate_polycubes(max_size=5):
    """Enumerate all free polycubes (distinct under rotation) of sizes 1..max_size.

    Uses canonical augmentation: grow from smaller polycubes by adding one cell
    at a time, then deduplicate using canonical (rotation-normalized) forms.

    Returns:
        dict mapping size -> list of polycubes (each a frozenset of (x,y,z) tuples)
    """
    catalog = {}

    # Size 1: single cube
    base = frozenset([(0, 0, 0)])
    catalog[1] = [base]

    for size in range(2, max_size + 1):
        seen_canonical = set()
        new_polycubes = []

        for poly in catalog[size - 1]:
            # Find all cells adjacent to the current polycube
            adj = set()
            for cell in poly:
                for nb in _neighbors_3d(cell):
                    if nb not in poly:
                        adj.add(nb)

            # Try adding each adjacent cell
            for cell in adj:
                candidate = poly | {cell}
                # Normalize and check all 24 rotations for canonical form
                canon = _canonical_form(candidate)
                if canon not in seen_canonical:
                    seen_canonical.add(canon)
                    new_polycubes.append(normalize(candidate))

        catalog[size] = new_polycubes

    return catalog


def _canonical_form(coords):
    """Get the canonical (minimum) form of a polycube across all 24 rotations.

    This is used for deduplication: two polycubes are the same free polycube
    if and only if they have the same canonical form.
    """
    from phase1.polycube import ROTATIONS, rotate

    forms = []
    for R in ROTATIONS:
        rotated = rotate(coords, R)
        normed = normalize(rotated)
        forms.append(normed)
    return min(forms, key=lambda s: tuple(sorted(s)))


def _find_one_instance_worker(args):
    """Find one solvable puzzle instance. Top-level so multiprocessing can pickle it."""
    available_pieces, target_volume, min_piece_size, max_piece_size, grid_size, dlx_timeout, seed = args
    random.seed(seed)
    for _ in range(300):
        pieces = _sample_piece_set(available_pieces, target_volume, min_piece_size, max_piece_size)
        if pieces is None:
            continue
        try:
            solutions = _solve_with_timeout(pieces, grid_size, dlx_timeout)
        except TimeoutError:
            continue
        if solutions:
            return {'pieces': pieces, 'grid_size': grid_size, 'solution': solutions[0]}
    return None


# ── Puzzle Instance Generation ────────────────────────────────────────────────

def generate_puzzle_instances(num_instances, grid_size, polycube_catalog,
                              min_piece_size=3, max_piece_size=5,
                              dlx_timeout=10.0, verbose=True, num_workers=1):
    """Generate solvable puzzle instances by random sampling + DLX verification.

    Strategy: randomly sample piece sets whose total volume = grid_size^3,
    then use DLX to check solvability. Keep only solvable instances.

    Args:
        num_instances: target number of solvable instances to generate
        grid_size: side length of target cube (e.g. 3 for 3x3x3)
        polycube_catalog: dict from enumerate_polycubes()
        min_piece_size: minimum piece size to use
        max_piece_size: maximum piece size to use
        dlx_timeout: max seconds for DLX per instance (skip if exceeded)
        verbose: print progress

    Returns:
        list of dicts: {
            'pieces': list of pieces (each a list of tuples),
            'grid_size': int,
            'solution': dict mapping piece_idx -> frozenset of cells
        }
    """
    target_volume = grid_size ** 3
    available_pieces = []
    for size in range(min_piece_size, max_piece_size + 1):
        if size in polycube_catalog:
            for poly in polycube_catalog[size]:
                available_pieces.append(list(poly))

    if not available_pieces:
        raise ValueError(f"No polycubes of size {min_piece_size}-{max_piece_size} in catalog")

    instances = []

    if num_workers > 1:
        import os
        from concurrent.futures import ProcessPoolExecutor
        num_workers = min(num_workers, os.cpu_count() or 1)
        n_tasks = max(num_instances * 4, num_workers * 10)
        base_seed = random.randint(0, 2 ** 20)
        task_args = [
            (available_pieces, target_volume, min_piece_size, max_piece_size,
             grid_size, dlx_timeout, base_seed + i)
            for i in range(n_tasks)
        ]
        chunk = num_workers * 4
        if verbose:
            print(f"  Parallel instance generation with {num_workers} workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for off in range(0, n_tasks, chunk):
                if len(instances) >= num_instances:
                    break
                for result in executor.map(_find_one_instance_worker,
                                           task_args[off:off + chunk]):
                    if result is not None and len(instances) < num_instances:
                        result['instance_id'] = len(instances)
                        instances.append(result)
                        if verbose:
                            print(f"  Instance {len(instances)}/{num_instances} found")
                    if len(instances) >= num_instances:
                        break
        if len(instances) < num_instances:
            print(f"Warning: only found {len(instances)}/{num_instances} instances")
        if verbose:
            print(f"Generated {len(instances)} instances with {num_workers} workers")
        return instances[:num_instances]

    attempts = 0
    max_total_attempts = num_instances * 500  # prevent infinite loop

    while len(instances) < num_instances and attempts < max_total_attempts:
        attempts += 1
        # Randomly select pieces that sum to target volume
        pieces = _sample_piece_set(available_pieces, target_volume,
                                   min_piece_size, max_piece_size)
        if pieces is None:
            continue

        # Try to solve with DLX (with timeout)
        t0 = time.time()
        try:
            solutions = _solve_with_timeout(pieces, grid_size, dlx_timeout)
        except TimeoutError:
            if verbose and attempts % 100 == 0:
                print(f"  Attempt {attempts}: timeout")
            continue

        elapsed = time.time() - t0

        if solutions:
            instances.append({
                'pieces': pieces,
                'grid_size': grid_size,
                'solution': solutions[0],
                'instance_id': len(instances),
            })
            if verbose:
                print(f"  Instance {len(instances)}/{num_instances} found "
                      f"(attempt {attempts}, {len(pieces)} pieces, {elapsed:.2f}s)")

        if verbose and attempts % 200 == 0 and not solutions:
            print(f"  Attempt {attempts}: {len(instances)}/{num_instances} found so far")

    if verbose:
        print(f"Generated {len(instances)} instances in {attempts} attempts")

    if len(instances) < num_instances:
        print(f"Warning: only found {len(instances)}/{num_instances} solvable instances "
              f"after {max_total_attempts} attempts")

    return instances


def _sample_piece_set(available_pieces, target_volume, min_size, max_size):
    """Randomly select pieces that sum to exactly target_volume.

    Uses a greedy approach with random shuffling: pick random pieces until
    we hit the target or overshoot.
    """
    pieces = []
    remaining = target_volume

    max_attempts = 100
    for _ in range(max_attempts):
        # Filter pieces that fit in the remaining volume
        valid = [p for p in available_pieces if len(p) <= remaining]
        if not valid:
            break

        piece = random.choice(valid)
        pieces.append(piece)
        remaining -= len(piece)

        if remaining == 0:
            return pieces

    return None  # Couldn't hit exact target


def _solve_with_timeout(pieces, grid_size, timeout):
    """Solve with DLX, but use a simple time check.

    Note: True preemptive timeout would require threading. We use a simpler
    approach here — we let DLX run and check time after. For truly large
    instances, consider using signal-based timeout on Unix.
    """
    import signal

    def _handler(signum, frame):
        raise TimeoutError("DLX solve timed out")

    # Use SIGALRM on Unix, fallback to no timeout on Windows
    use_alarm = hasattr(signal, 'SIGALRM')
    if use_alarm:
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(timeout))

    try:
        solutions = solve(pieces, grid_size=grid_size, find_all=False)
    except TimeoutError:
        raise
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    return solutions


# ── State Encoding ────────────────────────────────────────────────────────────

def encode_grid(solution_so_far, grid_size):
    """Encode a partial grid state as a binary 3D tensor.

    Args:
        solution_so_far: dict mapping piece_idx -> frozenset of (x,y,z) cells
        grid_size: int

    Returns:
        np.array of shape (grid_size, grid_size, grid_size), dtype float32
        1.0 = filled, 0.0 = empty
    """
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    for cells in solution_so_far.values():
        for x, y, z in cells:
            grid[x, y, z] = 1.0
    return grid


def encode_piece(piece, grid_size):
    """Encode a single piece as a 3D binary tensor (canonical shape padded to grid_size).

    The piece is normalized (origin at 0,0,0) and placed in a grid_size^3 tensor.

    Args:
        piece: list of (x,y,z) tuples
        grid_size: int

    Returns:
        np.array of shape (grid_size, grid_size, grid_size), dtype float32
    """
    tensor = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    normed = normalize(piece)
    for x, y, z in normed:
        if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
            tensor[x, y, z] = 1.0
    return tensor


def encode_remaining_pieces(pieces, grid_size, max_pieces):
    """Encode remaining pieces as a multi-channel 3D tensor.

    Args:
        pieces: list of pieces (each a list of (x,y,z) tuples)
        grid_size: int
        max_pieces: maximum number of piece channels (pad with zeros)

    Returns:
        np.array of shape (max_pieces, grid_size, grid_size, grid_size), dtype float32
    """
    channels = np.zeros((max_pieces, grid_size, grid_size, grid_size), dtype=np.float32)
    for i, piece in enumerate(pieces):
        if i >= max_pieces:
            break
        channels[i] = encode_piece(piece, grid_size)
    return channels


def encode_state(grid_state, remaining_pieces, grid_size, max_pieces):
    """Encode a full state (grid + remaining pieces) as the network input tensor.

    Returns:
        np.array of shape (1 + max_pieces, grid_size, grid_size, grid_size)
        Channel 0: grid occupancy
        Channels 1..max_pieces: one per remaining piece (canonical shape)
    """
    grid = grid_state if isinstance(grid_state, np.ndarray) else \
        encode_grid(grid_state, grid_size)
    pieces_enc = encode_remaining_pieces(remaining_pieces, grid_size, max_pieces)
    return np.concatenate([grid[np.newaxis], pieces_enc], axis=0)


def encode_placement(placement, grid_size):
    """Encode a placement (frozenset of cells) as a binary 3D tensor.

    Args:
        placement: frozenset of (x,y,z) tuples
        grid_size: int

    Returns:
        np.array of shape (grid_size, grid_size, grid_size), dtype float32
    """
    tensor = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    for x, y, z in placement:
        tensor[x, y, z] = 1.0
    return tensor


def _placement_sort_key(placement):
    """Deterministic key for ordering placements."""
    return tuple(sorted(placement))


def encode_policy_action_mask(placement, grid_size):
    """Encode a placement as a flat action mask used for policy supervision.

    The mask is normalized by piece size so dot products with cell logits
    represent the average score over the placement cells.
    """
    n3 = grid_size ** 3
    vec = np.zeros((n3,), dtype=np.float32)
    if not placement:
        return vec
    weight = 1.0 / float(len(placement))
    n2 = grid_size * grid_size
    for x, y, z in placement:
        vec[x * n2 + y * grid_size + z] = weight
    return vec


def build_policy_candidates(grid, next_piece, target_placement, grid_size):
    """Build valid placement actions and target index for policy training."""
    occupied = set(tuple(idx) for idx in np.argwhere(grid > 0.5))
    valid = [
        p for p in get_all_placements(next_piece, grid_size)
        if not (p & occupied)
    ]
    valid = sorted(valid, key=_placement_sort_key)

    if not valid:
        return np.zeros((0, grid_size ** 3), dtype=np.float32), -1

    action_masks = np.stack(
        [encode_policy_action_mask(p, grid_size) for p in valid], axis=0
    ).astype(np.float32)

    target_idx = -1
    for i, p in enumerate(valid):
        if p == target_placement:
            target_idx = i
            break

    return action_masks, target_idx


def _process_one_instance_worker(args):
    """Generate training examples for one instance. Top-level so multiprocessing can pickle it."""
    inst, inst_idx, max_pieces, num_negatives_per_solution = args
    pieces = inst['pieces']
    grid_size = inst['grid_size']
    solution = inst['solution']
    instance_id = inst.get('instance_id', inst_idx)

    examples = []
    piece_order = list(solution.keys())
    random.shuffle(piece_order)

    for step, piece_idx in enumerate(piece_order):
        remaining_indices = piece_order[step:]
        remaining_pieces_list = [pieces[i] for i in remaining_indices]
        placed_indices = piece_order[:step]
        placed_solution = {i: solution[i] for i in placed_indices}
        grid = encode_grid(placed_solution, grid_size)

        next_piece = remaining_pieces_list[0]
        next_placement = solution[piece_idx]
        policy_candidates, policy_target_idx = build_policy_candidates(
            grid, next_piece, next_placement, grid_size
        )
        if policy_target_idx < 0:
            continue

        state_tensor = encode_state(grid, remaining_pieces_list, grid_size, max_pieces)
        examples.append({
            'state': state_tensor,
            'grid': grid,
            'remaining_pieces': remaining_pieces_list,
            'label': 1.0,
            'value': len(remaining_indices),
            'next_placement': encode_placement(next_placement, grid_size),
            'next_piece_idx': 0,
            'policy_candidates': policy_candidates,
            'policy_target_idx': policy_target_idx,
            'grid_size': grid_size,
            'instance_id': instance_id,
        })

    neg_count = 0
    neg_attempts = 0
    max_neg_attempts = num_negatives_per_solution * 20
    while neg_count < num_negatives_per_solution and neg_attempts < max_neg_attempts:
        neg_attempts += 1
        neg_example = _generate_negative_example(pieces, grid_size, max_pieces,
                                                  instance_id=instance_id)
        if neg_example is not None:
            examples.append(neg_example)
            neg_count += 1

    return examples


# ── Training Data Generation ─────────────────────────────────────────────────

def generate_training_data(instances, max_pieces, num_negatives_per_solution=2,
                           verbose=True, num_workers=1):
    """Generate training examples from solved puzzle instances.

    From each solution, creates:
    - Positive examples: partial states from removing pieces one at a time
      (these are known solvable since they come from a valid solution)
    - Negative examples: randomly placed subsets that DLX confirms are unsolvable

    Args:
        instances: list of dicts from generate_puzzle_instances()
        max_pieces: max piece channels for encoding
        num_negatives_per_solution: negative examples per positive solution
        verbose: print progress

    Returns:
        list of training example dicts
    """
    examples = []

    if num_workers > 1:
        import os
        from concurrent.futures import ProcessPoolExecutor
        num_workers = min(num_workers, os.cpu_count() or 1)
        task_args = [(inst, i, max_pieces, num_negatives_per_solution)
                     for i, inst in enumerate(instances)]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for inst_idx, inst_examples in enumerate(
                    executor.map(_process_one_instance_worker, task_args, chunksize=4)):
                examples.extend(inst_examples)
                if verbose and (inst_idx + 1) % 10 == 0:
                    print(f"  Processed {inst_idx + 1}/{len(instances)} instances, "
                          f"{len(examples)} total examples")
        if verbose:
            pos_count = sum(1 for e in examples if e['label'] == 1.0)
            neg_count = sum(1 for e in examples if e['label'] == 0.0)
            print(f"Generated {len(examples)} examples "
                  f"({pos_count} positive, {neg_count} negative)")
        return examples

    for inst_idx, inst in enumerate(instances):
        pieces = inst['pieces']
        grid_size = inst['grid_size']
        solution = inst['solution']
        instance_id = inst.get('instance_id', inst_idx)
        num_pieces = len(pieces)

        # Generate positive examples: remove pieces one at a time from solution
        piece_order = list(solution.keys())
        random.shuffle(piece_order)

        # Start with full solution, remove pieces to create partial states
        partial_solution = dict(solution)  # copy

        for step, piece_idx in enumerate(piece_order):
            # The current state BEFORE removing this piece
            remaining_indices = piece_order[step:]
            remaining_pieces_list = [pieces[i] for i in remaining_indices]
            placed_indices = piece_order[:step]

            # Build grid from placed pieces only
            placed_solution = {i: solution[i] for i in placed_indices}
            grid = encode_grid(placed_solution, grid_size)

            # The next move: place the next piece
            next_piece_idx_in_remaining = 0  # first remaining piece
            next_piece = remaining_pieces_list[next_piece_idx_in_remaining]
            next_placement = solution[piece_idx]
            policy_candidates, policy_target_idx = build_policy_candidates(
                grid, next_piece, next_placement, grid_size
            )
            if policy_target_idx < 0:
                # Defensive guard: skip malformed positive examples.
                continue

            state_tensor = encode_state(
                grid, remaining_pieces_list, grid_size, max_pieces
            )

            examples.append({
                'state': state_tensor,
                'grid': grid,
                'remaining_pieces': remaining_pieces_list,
                'label': 1.0,  # solvable
                'value': len(remaining_indices),  # pieces remaining
                'next_placement': encode_placement(next_placement, grid_size),
                'next_piece_idx': next_piece_idx_in_remaining,
                'policy_candidates': policy_candidates,
                'policy_target_idx': policy_target_idx,
                'grid_size': grid_size,
                'instance_id': instance_id,
            })

        # Generate negative examples
        neg_count = 0
        neg_attempts = 0
        max_neg_attempts = num_negatives_per_solution * 20

        while neg_count < num_negatives_per_solution and neg_attempts < max_neg_attempts:
            neg_attempts += 1
            neg_example = _generate_negative_example(
                pieces, grid_size, max_pieces, instance_id=instance_id
            )
            if neg_example is not None:
                examples.append(neg_example)
                neg_count += 1

        if verbose and (inst_idx + 1) % 10 == 0:
            print(f"  Processed {inst_idx + 1}/{len(instances)} instances, "
                  f"{len(examples)} total examples")

    if verbose:
        pos_count = sum(1 for e in examples if e['label'] == 1.0)
        neg_count = sum(1 for e in examples if e['label'] == 0.0)
        print(f"Generated {len(examples)} examples "
              f"({pos_count} positive, {neg_count} negative)")

    return examples


def _generate_negative_example(pieces, grid_size, max_pieces, instance_id=None):
    """Try to create an unsolvable partial state.

    Strategy: randomly place a subset of pieces in valid positions, then
    check if the remaining pieces can fill the gaps using DLX.
    If DLX says no, we have a negative example.
    """
    num_pieces = len(pieces)
    if num_pieces < 2:
        return None

    # Pick a random number of pieces to place (1 to num_pieces-1)
    num_to_place = random.randint(1, num_pieces - 1)

    # Randomly order pieces
    indices = list(range(num_pieces))
    random.shuffle(indices)

    placed_indices = indices[:num_to_place]
    remaining_indices = indices[num_to_place:]

    # Try to place the selected pieces randomly in the grid
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    placed_solution = {}
    occupied = set()

    for idx in placed_indices:
        piece = pieces[idx]
        placements = get_all_placements(piece, grid_size)
        # Filter to non-overlapping placements
        valid = [p for p in placements if not (p & occupied)]
        if not valid:
            return None  # Can't place this piece, abort
        placement = random.choice(valid)
        placed_solution[idx] = placement
        occupied |= placement
        for x, y, z in placement:
            grid[x, y, z] = 1.0

    # Check if remaining pieces can fill the rest using DLX
    remaining_pieces = [pieces[i] for i in remaining_indices]
    empty_cells = set()
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                if (x, y, z) not in occupied:
                    empty_cells.add((x, y, z))

    # Quick volume check
    remaining_volume = sum(len(p) for p in remaining_pieces)
    if remaining_volume != len(empty_cells):
        # Volume mismatch — trivially unsolvable
        # But this is too easy for the network, so skip it
        return None

    # Use DLX to check solvability of remaining pieces in empty space
    solvable = _check_partial_solvability(
        remaining_pieces, empty_cells, grid_size
    )

    if solvable:
        return None  # This turned out solvable — not a negative example

    # Confirmed unsolvable! Create the example
    remaining_pieces_list = [pieces[i] for i in remaining_indices]
    state_tensor = encode_state(grid, remaining_pieces_list, grid_size, max_pieces)

    return {
        'state': state_tensor,
        'grid': grid,
        'remaining_pieces': remaining_pieces_list,
        'label': 0.0,  # unsolvable
        'value': len(remaining_indices),
        'next_placement': np.zeros((grid_size, grid_size, grid_size), dtype=np.float32),
        'next_piece_idx': -1,  # no valid next move
        'policy_candidates': np.zeros((0, grid_size ** 3), dtype=np.float32),
        'policy_target_idx': -1,
        'grid_size': grid_size,
        'instance_id': instance_id,
    }


def _check_partial_solvability(pieces, empty_cells, grid_size):
    """Check if pieces can exactly fill the given empty cells using DLX.

    This is a constrained version of the full solver: pieces must cover
    exactly the empty_cells and nothing else.
    """
    if not pieces or not empty_cells:
        return len(pieces) == 0 and len(empty_cells) == 0

    # Column names: one per empty cell + one per piece
    cell_names = [f"c_{x}_{y}_{z}" for x, y, z in sorted(empty_cells)]
    piece_names = [f"p_{i}" for i in range(len(pieces))]
    all_columns = cell_names + piece_names

    dlx = DLX(all_columns)

    row_id = 0
    for piece_idx, piece in enumerate(pieces):
        placements = get_all_placements(piece, grid_size)
        has_valid = False
        for placement in placements:
            # Only consider placements entirely within empty cells
            if placement.issubset(empty_cells):
                cols = [f"p_{piece_idx}"]
                for x, y, z in placement:
                    cols.append(f"c_{x}_{y}_{z}")
                dlx.add_row(row_id, cols)
                row_id += 1
                has_valid = True

        if not has_valid:
            return False  # This piece can't be placed at all

    solutions = dlx.solve(find_all=False)
    return len(solutions) > 0


# ── Dataset Utilities ─────────────────────────────────────────────────────────

def create_torch_dataset(examples):
    """Convert list of example dicts to a PyTorch Dataset.

    Args:
        examples: list of dicts from generate_training_data()

    Returns:
        PolycubeDataset instance
    """
    import torch
    from torch.utils.data import Dataset

    class PolycubeDataset(Dataset):
        def __init__(self, examples):
            if not examples:
                raise ValueError("examples must be non-empty")

            self.states = torch.tensor(
                np.array([e['state'] for e in examples]), dtype=torch.float32
            )
            self.labels = torch.tensor(
                np.array([e['label'] for e in examples]), dtype=torch.float32
            )
            self.values = torch.tensor(
                np.array([e['value'] for e in examples]), dtype=torch.float32
            )
            self.next_placements = torch.tensor(
                np.array([e['next_placement'] for e in examples]), dtype=torch.float32
            )
            self.next_piece_indices = torch.tensor(
                np.array([e['next_piece_idx'] for e in examples]), dtype=torch.long
            )

            # Build placement-action candidates for policy training.
            n_examples = len(examples)
            n3 = int(np.prod(examples[0]['state'].shape[1:]))

            candidate_lists = []
            target_indices = []
            for e in examples:
                cands = e.get('policy_candidates', None)
                tgt = e.get('policy_target_idx', None)

                if cands is None:
                    # Backward compatibility with older example format.
                    if e.get('next_piece_idx', -1) >= 0:
                        flat = np.array(e['next_placement'], dtype=np.float32).reshape(-1)
                        s = float(flat.sum())
                        if s > 0:
                            flat = flat / s
                        cands = flat.reshape(1, -1)
                        tgt = 0
                    else:
                        cands = np.zeros((0, n3), dtype=np.float32)
                        tgt = -1

                cands = np.array(cands, dtype=np.float32)
                if cands.ndim == 1:
                    cands = cands.reshape(1, -1)
                if cands.shape[0] == 0:
                    cands = np.zeros((0, n3), dtype=np.float32)
                if cands.shape[-1] != n3:
                    raise ValueError(
                        f"policy candidate width {cands.shape[-1]} != state width {n3}"
                    )
                if tgt is None:
                    tgt = -1

                candidate_lists.append(cands)
                target_indices.append(int(tgt))

            max_actions = max(c.shape[0] for c in candidate_lists)
            max_actions = max(max_actions, 1)
            policy_candidates = np.zeros(
                (n_examples, max_actions, n3), dtype=np.float32
            )
            policy_action_mask = np.zeros(
                (n_examples, max_actions), dtype=np.bool_
            )
            for i, cands in enumerate(candidate_lists):
                n = cands.shape[0]
                if n > 0:
                    policy_candidates[i, :n, :] = cands
                    policy_action_mask[i, :n] = True

            self.policy_candidates = torch.tensor(policy_candidates, dtype=torch.float32)
            self.policy_action_mask = torch.tensor(policy_action_mask, dtype=torch.bool)
            self.policy_target_idx = torch.tensor(
                np.array(target_indices, dtype=np.int64), dtype=torch.long
            )

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                'state': self.states[idx],
                'label': self.labels[idx],
                'value': self.values[idx],
                'next_placement': self.next_placements[idx],
                'next_piece_idx': self.next_piece_indices[idx],
                'policy_candidates': self.policy_candidates[idx],
                'policy_action_mask': self.policy_action_mask[idx],
                'policy_target_idx': self.policy_target_idx[idx],
            }

    return PolycubeDataset(examples)


def split_dataset(examples, val_fraction=0.15, seed=42, group_key='instance_id'):
    """Split examples into train and validation sets.

    Args:
        examples: list of example dicts
        val_fraction: fraction for validation
        seed: random seed for reproducibility
        group_key: if present in examples, split by group id to avoid leakage

    Returns:
        (train_examples, val_examples)
    """
    if not examples:
        return [], []

    rng = random.Random(seed)
    use_grouped_split = (
        group_key is not None and any(group_key in ex for ex in examples)
    )

    if not use_grouped_split:
        shuffled = list(examples)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - val_fraction))
        return shuffled[:split_idx], shuffled[split_idx:]

    # Group-aware split to prevent near-duplicate states from the same instance
    # leaking across train/val.
    groups = {}
    for idx, ex in enumerate(examples):
        gid = ex.get(group_key, f"__ungrouped_{idx}")
        groups.setdefault(gid, []).append(ex)

    group_ids = list(groups.keys())
    rng.shuffle(group_ids)

    target_val = int(len(examples) * val_fraction)
    val_examples = []
    train_examples = []
    for gid in group_ids:
        bucket = groups[gid]
        if len(val_examples) < target_val:
            val_examples.extend(bucket)
        else:
            train_examples.extend(bucket)

    # Guard against degenerate tiny datasets.
    if not val_examples and train_examples:
        val_examples.append(train_examples.pop())
    if not train_examples and val_examples:
        train_examples.append(val_examples.pop())

    return train_examples, val_examples


# ── Quick-generation helper ───────────────────────────────────────────────────

def generate_soma_training_data(num_instances=100, num_negatives=2, max_pieces=10):
    """Convenience function to generate training data from Soma cube variations.

    Uses the 7 Soma pieces (which always solve 3x3x3) as a fast source
    of training data — no need to search for solvable instances.

    Args:
        num_instances: how many random orderings/decompositions to generate
        num_negatives: negative examples per solution
        max_pieces: max piece channels for encoding

    Returns:
        list of training example dicts
    """
    from phase1.test_cases import SOMA_PIECES

    print(f"Generating Soma training data ({num_instances} instances)...")

    # Get all solutions (DLX is fast for Soma) for data diversity
    solutions = solve(SOMA_PIECES, grid_size=3, find_all=True)
    if not solutions:
        raise RuntimeError("Could not solve Soma cube")

    print(f"  Found {len(solutions)} distinct Soma solutions for data diversity")

    # Create instances cycling through different solutions
    instances = []
    for i in range(num_instances):
        instances.append({
            'pieces': SOMA_PIECES,
            'grid_size': 3,
            'solution': solutions[i % len(solutions)],
            'instance_id': i,
        })

    examples = generate_training_data(
        instances, max_pieces=max_pieces,
        num_negatives_per_solution=num_negatives, verbose=True
    )
    return examples


if __name__ == "__main__":
    # Quick test: enumerate polycubes and generate some training data
    print("Enumerating polycubes...")
    catalog = enumerate_polycubes(max_size=5)
    for size, polys in sorted(catalog.items()):
        print(f"  Size {size}: {len(polys)} free polycubes")

    print("\nGenerating Soma training data...")
    examples = generate_soma_training_data(num_instances=20, num_negatives=2)
    print(f"\nTotal examples: {len(examples)}")
    print(f"  Positive: {sum(1 for e in examples if e['label'] == 1.0)}")
    print(f"  Negative: {sum(1 for e in examples if e['label'] == 0.0)}")

    if examples:
        ex = examples[0]
        print(f"\nExample state shape: {ex['state'].shape}")
        print(f"  Label: {ex['label']}, Value: {ex['value']}")
