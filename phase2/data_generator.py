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


# ── Puzzle Instance Generation ────────────────────────────────────────────────

def generate_puzzle_instances(num_instances, grid_size, polycube_catalog,
                              min_piece_size=3, max_piece_size=5,
                              dlx_timeout=10.0, verbose=True):
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


# ── Training Data Generation ─────────────────────────────────────────────────

def generate_training_data(instances, max_pieces, num_negatives_per_solution=2,
                           verbose=True):
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

    for inst_idx, inst in enumerate(instances):
        pieces = inst['pieces']
        grid_size = inst['grid_size']
        solution = inst['solution']
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
            next_placement = solution[piece_idx]

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
                'next_piece_idx': 0,  # index into remaining pieces list
                'grid_size': grid_size,
            })

        # Generate negative examples
        neg_count = 0
        neg_attempts = 0
        max_neg_attempts = num_negatives_per_solution * 20

        while neg_count < num_negatives_per_solution and neg_attempts < max_neg_attempts:
            neg_attempts += 1
            neg_example = _generate_negative_example(
                pieces, grid_size, max_pieces
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


def _generate_negative_example(pieces, grid_size, max_pieces):
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
        'grid_size': grid_size,
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

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                'state': self.states[idx],
                'label': self.labels[idx],
                'value': self.values[idx],
                'next_placement': self.next_placements[idx],
                'next_piece_idx': self.next_piece_indices[idx],
            }

    return PolycubeDataset(examples)


def split_dataset(examples, val_fraction=0.15, seed=42):
    """Split examples into train and validation sets.

    Args:
        examples: list of example dicts
        val_fraction: fraction for validation
        seed: random seed for reproducibility

    Returns:
        (train_examples, val_examples)
    """
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_fraction))
    return shuffled[:split_idx], shuffled[split_idx:]


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
