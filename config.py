"""
Tunable configuration for the neural polycube solver.

Edit these parameters to experiment with different settings.
Import this file in notebooks or scripts:

    from config import CONFIG
    model = create_model(**CONFIG['model'])
"""

CONFIG = {

    # ── Puzzle Settings ──────────────────────────────────────────────────────
    'puzzle': {
        'grid_size': 3,             # Target cube: 3 → 3x3x3 (27 cells), 4 → 4x4x4 (64 cells)
        'max_pieces': 10,           # Max piece channels in state encoding
                                    #   Must be >= number of pieces in your puzzle
                                    #   Soma has 7; larger puzzles may need more
    },

    # ── Neural Network Architecture ──────────────────────────────────────────
    # These define the CuboidNet model structure.
    # Changing these requires retraining from scratch.
    'model': {
        'grid_size': 3,             # Must match puzzle.grid_size
        'max_pieces': 10,           # Must match puzzle.max_pieces
        'num_residual_blocks': 6,   # Depth of the ResNet (try 4 for faster, 8 for more capacity)
        'hidden_dim': 128,          # Width of hidden layers (try 64 for lighter, 256 for heavier)
    },

    # ── Training Data Generation ─────────────────────────────────────────────
    'data': {
        'num_instances': 100,       # Number of puzzle instances to generate
                                    #   More = better coverage but slower generation
                                    #   For 3x3x3 Soma: 100 → ~900 examples
                                    #   For 4x4x4: start with 20-50 (slower DLX)
        'num_negatives': 2,         # Negative examples per solution
                                    #   Higher = more balanced dataset
        'val_fraction': 0.15,       # Fraction held out for validation
        'polycube_max_size': 5,     # Max polycube size for enumeration (3-5)
                                    #   Size 5 has 29 shapes; size 6 has 166 (much slower)
        'dlx_timeout': 10.0,        # Seconds before giving up on a DLX solve during data gen
                                    #   Increase for 4x4x4 (try 30-60)
    },

    # ── Supervised Training ──────────────────────────────────────────────────
    'train': {
        'epochs': 50,               # Training epochs (30-100 typical)
        'lr': 1e-3,                 # Learning rate (Adam optimizer)
        'batch_size': 64,           # Batch size (reduce to 32 if memory issues)
        'lambda_value': 1.0,        # Weight for value (solvability) loss
        'lambda_policy': 0.0,       # Weight for policy (placement) loss
                                    #   Current best on harder benchmark: keep value-only default
        'weight_decay': 1e-4,       # L2 regularization
        'grad_clip': 1.0,           # Gradient norm clipping
    },

    # ── Autodidactic Iteration (ADI) ─────────────────────────────────────────
    # Self-play improvement loop. Run after supervised training.
    'adi': {
        'num_rounds': 2,            # Number of ADI iterations (1-3 typical)
        'num_new_instances': 30,    # Beam searches per round
        'beam_width': 8,            # Narrow beam to generate failures for learning
                                    #   Must be small enough that some searches fail
        'adi_epochs': 15,           # Retraining epochs per round
        'lr': 5e-4,                 # Lower LR for fine-tuning (round 1)
        'lr_decay': 0.6,            # Multiply LR by this each round (round 2 = 3e-4)
        'timeout': 15.0,            # Per-instance beam search timeout (seconds)
    },

    # ── NN Beam Search Solver ────────────────────────────────────────────────
    # These affect solve speed and success rate at inference time.
    'solver': {
        'beam_width': 32,           # States kept per depth level
                                    #   8  → fast, ~45-65% solve rate on Soma
                                    #   32 → moderate, ~100% on Soma
                                    #   64 → thorough but slower
                                    #  128 → for harder puzzles
        'timeout': 30.0,            # Max seconds for NN search before giving up
        'max_candidates_per_state': 50,  # Cap child states per beam state
                                    #   Prevents memory explosion on large puzzles
    },

    # ── Hybrid Solver ────────────────────────────────────────────────────────
    'hybrid': {
        'beam_width': 64,           # Wider beam for production use
        'timeout_nn': 30,           # Seconds for NN attempt
        'timeout_dlx': 120,         # Seconds for DLX fallback (Unix only)
        'model_name': 'soma_3x3x3',  # Trained model to load
                                    #   After ADI: try 'soma_3x3x3_adi2'
    },

    # ── Device ───────────────────────────────────────────────────────────────
    'device': 'cpu',                # 'cpu' or 'cuda' (if GPU available)
}


# ── Current Test Cases ───────────────────────────────────────────────────────
#
# TESTED:
#   - 3x3x3 Soma cube (7 pieces: 1 tricube + 6 tetracubes, 27 cells)
#     Model: soma_3x3x3.pt — 91.9% val accuracy, 100% solve rate (bw=32)
#
# NOT YET TESTED (infrastructure exists, needs running):
#   - 4x4x4 cube (64 cells) — requires random piece sets from polycube catalog
#     To try: run_supervised_training(grid_size=4, num_instances=20, dlx_timeout=30)
#     Expected: slower data gen, may need more epochs and wider beam
#
#   - 5x5x5 cube (125 cells) — see "Scaling to Larger Cubes" below
#
#
# ── Scaling to Larger Cubes (4x4x4, 5x5x5, and beyond) ─────────────────────
#
# The current bottleneck for large cubes is TRAINING DATA GENERATION, not the
# NN solver itself. DLX becomes too slow to verify solvability for 5x5x5+.
# Three strategies to break this dependency:
#
# STRATEGY 1: Constructive Data Generation (no DLX needed)
#   Instead of randomly sampling pieces and hoping DLX can solve them,
#   BUILD solutions by construction:
#   - Start with an empty NxNxN cube
#   - Randomly partition it into connected polycube-shaped regions
#   - Each partition is a guaranteed-solvable puzzle instance
#   This completely bypasses DLX for data generation and works at any scale.
#   Implementation: write a random_partition(grid_size, piece_sizes) function
#   that flood-fills the cube into valid polycube pieces.
#
# STRATEGY 2: Grid-Size Agnostic Architecture
#   The current CuboidNet has FC layers tied to grid_size:
#     FC(32 * N^3, 256) in the value head
#     FC(64 * N^3, 512) in the policy head
#   This means a model trained on 3x3x3 CANNOT be used on 5x5x5.
#   Fix: replace FC layers with Global Average Pooling (GAP):
#     Conv3d(hidden, 32, 1) → BatchNorm → ReLU → GlobalAvgPool3d → FC(32, 1)
#   Then the same model works on ANY grid size. Train on mixed 3x3x3 + 4x4x4
#   data, and the learned spatial features transfer to 5x5x5 at test time.
#
# STRATEGY 3: Curriculum Learning + ADI Bootstrap
#   Train progressively on harder puzzles:
#     Phase A: Supervised training on 3x3x3 (fast DLX, many examples)
#     Phase B: Fine-tune on 4x4x4 (slower DLX, fewer examples)
#     Phase C: ADI-only on 5x5x5 (no DLX needed!)
#       - Use constructive data gen (Strategy 1) for initial puzzle instances
#       - NN solver attempts them with beam search
#       - Successes → positive training data, failures → negative
#       - Retrain and repeat — the model bootstraps itself
#   This is closest to the original DeepCube paper's approach.
#
# RECOMMENDED ORDER OF IMPLEMENTATION:
#   1. Constructive data gen (Strategy 1) — unblocks everything else
#   2. Grid-agnostic architecture (Strategy 2) — enables transfer learning
#   3. Curriculum + ADI (Strategy 3) — ties it all together
#
#
# ── Scaling Analysis: How Large Can We Go? ───────────────────────────────────
#
# Grid    Cells   ~Pieces   Input Tensor Shape      DLX Feasible?   Status
# -----   -----   -------   ----------------------  -------------   ------
# 3x3x3      27     7       (11, 3, 3, 3)           Yes (ms)        DONE
# 4x4x4      64    ~15      (16, 4, 4, 4)           Yes (seconds)   Ready to run
# 5x5x5     125    ~30      (31, 5, 5, 5)           Marginal        Needs Strategies 1-3
# 6x6x6     216    ~50      (51, 6, 6, 6)           No              Needs arch changes
# 10x10x10 1000   ~200+     (201, 10, 10, 10)       No              Research-level
#
# The GRID SIZE itself is not the bottleneck — 3D convolutions on 10x10x10 are
# fine (small by image standards). The real scaling challenges are:
#
# CHALLENGE 1: Piece Encoding Explosion
#   Current approach: one 3D channel per remaining piece.
#   At 200+ pieces, the input tensor becomes ~200K floats per state.
#   Solutions:
#   - Piece histogram: encode a count vector of piece sizes instead of
#     individual shapes. E.g., "12 tricubes, 45 tetracubes, 30 pentacubes"
#     as a small fixed-size vector, independent of piece count.
#   - Attention-based piece encoder: process pieces as a variable-length
#     sequence with a Transformer/attention layer, producing a fixed-size
#     summary embedding. Concatenate with the grid convolution features.
#   - Top-K encoding: only encode the K most constrained remaining pieces
#     (by MRV count), not all of them. The model doesn't need to "see"
#     pieces with many placement options.
#
# CHALLENGE 2: Branching Factor
#   Each piece has more valid placements in a larger grid.
#   A tetracube in 3x3x3 has ~100 placements; in 10x10x10 it has ~10,000+.
#   Solutions:
#   - Policy head becomes critical: use it to rank/filter placements
#     before expanding, rather than trying all of them.
#   - Spatial locality: only consider placements near existing pieces
#     or near corners/edges, pruning distant placements.
#   - Increase max_candidates_per_state (currently 50) but pair with
#     stronger policy filtering.
#
# CHALLENGE 3: Search Depth
#   200+ pieces = 200+ levels of beam search. Each level has NN inference.
#   Solutions:
#   - Batch multiple depth levels (place several pieces between NN evals).
#   - Hierarchical decomposition: divide 10x10x10 into 5x5x5 sub-regions,
#     assign pieces to sub-regions, solve each independently.
#   - Monte Carlo Tree Search (MCTS) instead of beam search — focuses
#     computation on the most promising branches rather than keeping a
#     fixed-width beam at every depth.
#
# REALISTIC ROADMAP:
#   Near-term:  5x5x5  (~30 pieces)  — achievable with Strategies 1-3 above
#   Mid-term:   7x7x7  (~70 pieces)  — needs piece encoding + policy changes
#   Long-term:  10x10x10 (200+ pieces) — needs hierarchical decomposition
#
#
# ── Quick Start ──────────────────────────────────────────────────────────────
#
# 1. Train a model:
#    from config import CONFIG
#    from phase2.train import run_supervised_training
#    model, history, examples = run_supervised_training(
#        grid_size=CONFIG['puzzle']['grid_size'],
#        max_pieces=CONFIG['puzzle']['max_pieces'],
#        num_instances=CONFIG['data']['num_instances'],
#        epochs=CONFIG['train']['epochs'],
#        lr=CONFIG['train']['lr'],
#        batch_size=CONFIG['train']['batch_size'],
#        hidden_dim=CONFIG['model']['hidden_dim'],
#        num_residual_blocks=CONFIG['model']['num_residual_blocks'],
#        save_name="my_model",
#    )
#
# 2. Run ADI:
#    from phase2.train import run_adi_iteration
#    model, adi_hist, adi_ex = run_adi_iteration(
#        model, grid_size=CONFIG['puzzle']['grid_size'],
#        max_pieces=CONFIG['puzzle']['max_pieces'],
#        num_new_instances=CONFIG['adi']['num_new_instances'],
#        beam_width=CONFIG['adi']['beam_width'],
#        adi_epochs=CONFIG['adi']['adi_epochs'],
#        lr=CONFIG['adi']['lr'],
#        existing_examples=examples,
#    )
#
# 3. Solve with the model:
#    from hybrid_solver import hybrid_solve
#    result = hybrid_solve(pieces, grid_size=3,
#                          model_name="my_model",
#                          beam_width=CONFIG['solver']['beam_width'])
