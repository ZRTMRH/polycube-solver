
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import torch

# Phase 1 imports
from phase1.polycube import get_orientations, get_all_placements
from phase1.solver import solve
from phase1.visualization import plot_solution, animate_solution, plot_pieces
from phase1.test_cases import (
    MONOMINO, DOMINO, L_TRICUBE, SOMA_PIECES,
    verify_solution
)

GRID_SIZE = 4
MAX_PIECES = 21  # 4x4x4 = 64 cells, worst case ~21 tricubes

# ── Phase 1: Basic validation (keep as-is, these are 2x2x2 and 3x3x3 demos) ──

orientations = get_orientations(L_TRICUBE)
print(f"L-tricube: {len(L_TRICUBE)} cubes, {len(orientations)} unique orientations")

pieces_2x2 = [DOMINO] * 4
solutions = solve(pieces_2x2)
if solutions:
    sol = solutions[0]
    assert verify_solution(sol, 2), "Solution verification failed!"
    print("Verified: 2x2x2 with dominoes.")
fig, ax = plot_solution(sol, 2, title="2×2×2 Cube — 4 Dominoes")
plt.show()

# ── Generate 4x4x4 piece sets using the same shapes as SOMA_PIECES ──────────
# Strategy: use the polycube catalog (same shapes as Soma) to find piece sets
# that sum to 64 cells and can fill a 4x4x4 cube

from phase2.data_generator import enumerate_polycubes, generate_puzzle_instances, generate_training_data

catalog = enumerate_polycubes(max_size=5)
total = 0
for size, polys in sorted(catalog.items()):
    print(f"Size {size}: {len(polys):3d} free polycubes")
    total += len(polys)
print(f"Total: {total} polycubes (sizes 1-5)")

# Generate solvable 4x4x4 instances (pieces drawn from same shapes as Soma)
instances = generate_puzzle_instances(
    num_instances=100,
    grid_size=GRID_SIZE,
    polycube_catalog=catalog,
    min_piece_size=3,
    max_piece_size=5,
    dlx_timeout=10.0,
    verbose=True
)

# Use first instance as our test case
test_instance = instances[0]
test_pieces = test_instance['pieces']
test_solution = test_instance['solution']

# Visualize the pieces we'll be working with
fig = plot_pieces(test_pieces, title=f"4x4x4 Test Pieces ({len(test_pieces)} pieces)")
plt.show()

# Solve with DLX to verify
dlx_sol = solve(test_pieces, grid_size=GRID_SIZE)
if dlx_sol:
    dlx_solution = dlx_sol[0]
    assert verify_solution(dlx_solution, GRID_SIZE), "DLX solution verification failed!"
    print("Verified: perfect 4x4x4 cube!")
    fig, ax = plot_solution(dlx_solution, GRID_SIZE, title="4x4x4 DLX Solution")
    plt.show()
    anim = animate_solution(dlx_solution, GRID_SIZE, title="4x4x4 Cube Assembly", interval=1000)
    plt.show()

# ── Generate Training Data ───────────────────────────────────────────────────

examples = generate_training_data(
    instances,
    max_pieces=MAX_PIECES,
    num_negatives_per_solution=2,
    verbose=True
)

pos = sum(1 for e in examples if e['label'] == 1.0)
neg = sum(1 for e in examples if e['label'] == 0.0)
print(f"\nDataset: {len(examples)} examples ({pos} positive, {neg} negative)")
print(f"State tensor shape: {examples[0]['state'].shape}")

# Visualize partial states — use actual piece counts from 4x4x4
num_pieces = len(test_pieces)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), subplot_kw={'projection': '3d'})
for ax_idx, pieces_remaining in enumerate([num_pieces - 1, num_pieces // 2, 1]):
    try:
        ex = next(e for e in examples if e['value'] == pieces_remaining and e['label'] == 1.0)
    except StopIteration:
        continue
    grid = ex['grid']
    filled = grid > 0.5
    ax = axes[ax_idx]
    import matplotlib.colors as mcolors
    facecolors = np.where(filled[..., np.newaxis],
                          np.array(mcolors.to_rgba('#4363d8', alpha=0.7)),
                          np.array([0, 0, 0, 0]))
    edgecolors = np.where(filled[..., np.newaxis],
                          np.array([0, 0, 0, 0.3]),
                          np.array([0, 0, 0, 0]))
    ax.voxels(filled, facecolors=facecolors, edgecolors=edgecolors)
    placed = num_pieces - pieces_remaining
    ax.set_title(f"{placed}/{num_pieces} pieces placed\n({pieces_remaining} remaining)")
    ax.set_xlim(0, GRID_SIZE); ax.set_ylim(0, GRID_SIZE); ax.set_zlim(0, GRID_SIZE)

plt.suptitle("Training Examples: Partial 4x4x4 Cube States", fontsize=14)
plt.tight_layout()
plt.show()

# ── Build and Train Model ────────────────────────────────────────────────────

from phase2.nn_model import create_model, model_summary
from phase2.data_generator import split_dataset, create_torch_dataset
from phase2.train import train, save_model, plot_training_curves
from torch.utils.data import DataLoader

model = create_model(grid_size=GRID_SIZE, max_pieces=MAX_PIECES,
                     num_residual_blocks=6, hidden_dim=128)
model_summary(model, grid_size=GRID_SIZE, max_pieces=MAX_PIECES)

train_ex, val_ex = split_dataset(examples)
print(f"Train: {len(train_ex)}, Val: {len(val_ex)}")

train_dataset = create_torch_dataset(train_ex)
val_dataset = create_torch_dataset(val_ex)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

history = train(model, train_loader, val_loader, epochs=50, lr=1e-3)

save_model(model, "4x4x4", history, metadata={
    'grid_size': GRID_SIZE, 'max_pieces': MAX_PIECES,
    'num_examples': len(examples), 'epochs': 50,
})

fig = plot_training_curves(history, title="CuboidNet Training (4x4x4 Cube)")
plt.show()

# ── NN Beam Search ───────────────────────────────────────────────────────────

from phase2.nn_solver import nn_solve

print("NN beam search on 4x4x4 cube...")
t0 = time.time()
nn_solution = nn_solve(test_pieces, grid_size=GRID_SIZE, model=model,
                       max_pieces=MAX_PIECES, beam_width=64, timeout=30.0)
nn_time = time.time() - t0

if nn_solution is not None:
    print(f"NN solved in {nn_time:.3f}s!")
    assert verify_solution(nn_solution, GRID_SIZE), "NN solution verification failed!"
    print("Verified: valid 4x4x4 cube!")
else:
    print(f"NN failed after {nn_time:.3f}s")

t0 = time.time()
dlx_sol = solve(test_pieces, grid_size=GRID_SIZE)
dlx_time = time.time() - t0
print(f"DLX solved in {dlx_time:.3f}s")

if nn_solution is not None:
    print(f"NN time: {nn_time:.3f}s vs DLX time: {dlx_time:.3f}s")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})
    plot_solution(nn_solution, GRID_SIZE, title="NN Solver Solution", ax=axes[0])
    plot_solution(dlx_sol[0], GRID_SIZE, title="DLX Solver Solution", ax=axes[1])
    plt.suptitle("4x4x4 Cube: NN vs DLX Solutions", fontsize=14)
    plt.tight_layout()
    plt.show()

# ── Confidence Tracing ───────────────────────────────────────────────────────

from phase2.data_generator import encode_state, encode_grid

dlx_solution = dlx_sol[0] if dlx_sol else test_solution
piece_order = sorted(dlx_solution.keys())
confidences = []
partial = {}

for step, pidx in enumerate(piece_order):
    remaining_indices = piece_order[step:]
    remaining_pieces = [test_pieces[i] for i in remaining_indices]
    grid = encode_grid(partial, GRID_SIZE)
    state = encode_state(grid, remaining_pieces, GRID_SIZE, MAX_PIECES)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        value, _ = model(state_tensor)
    confidences.append(value.item())
    partial[pidx] = dlx_solution[pidx]

confidences.append(1.0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(len(confidences)), confidences, 'o-', color='#4363d8', linewidth=2, markersize=8)
ax.set_xlabel('Pieces Placed')
ax.set_ylabel('P(solvable)')
ax.set_title('NN Confidence Along Solution Path (4x4x4)')
ax.set_xticks(range(len(confidences)))
ax.set_xticklabels([f'{i}\n({num_pieces-i} left)' for i in range(len(confidences))])
ax.set_ylim(-0.05, 1.05)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Decision boundary')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ── Hybrid Solver ────────────────────────────────────────────────────────────

from hybrid_solver import hybrid_solve
#increased the timeout_n from 30 to 60 and timeout_dlx from 120 to 200
result = hybrid_solve(test_pieces, grid_size=GRID_SIZE,
                      model_name="4x4x4", beam_width=64, timeout_nn=60, timeout_dlx=200)

print(f"\nResult: solved via '{result['method']}' in {result['time']:.3f}s")
if result['solution'] is not None:
    assert verify_solution(result['solution'], GRID_SIZE), "Hybrid solution verification failed!"
    print("Verified: valid 4x4x4 cube!")
    fig, ax = plot_solution(result['solution'], GRID_SIZE,
                            title=f"Hybrid Solver ({result['method'].upper()}) Solution")
    plt.show()

# ── ADI Training ─────────────────────────────────────────────────────────────

from phase2.train import run_adi_iteration

def measure_solve_rate(model, n_trials=20, beam_width=8, timeout=15.0):
    solved = 0
    for _ in range(n_trials):
        inst = random.choice(instances)
        pieces = inst['pieces']
        sol = nn_solve(pieces, GRID_SIZE, model, max_pieces=MAX_PIECES,
                       beam_width=beam_width, timeout=timeout)
        if sol is not None:
            solved += 1
    return solved / n_trials

print("Pre-ADI solve rate (beam_width=8, 20 trials)...")
pre_rate = measure_solve_rate(model, n_trials=20, beam_width=8)
print(f"  Solve rate: {pre_rate:.0%}")

model, adi_history_1, adi_examples_1 = run_adi_iteration(
    model, grid_size=GRID_SIZE, max_pieces=MAX_PIECES,
    num_new_instances=30, beam_width=8,
    adi_epochs=15, lr=5e-4, batch_size=64,
    existing_examples=examples,
)
save_model(model, "4x4x4_adi1", adi_history_1)
fig = plot_training_curves(adi_history_1, title="ADI Round 1 Training Curves")
plt.show()

model, adi_history_2, adi_examples_2 = run_adi_iteration(
    model, grid_size=GRID_SIZE, max_pieces=MAX_PIECES,
    num_new_instances=30, beam_width=8,
    adi_epochs=15, lr=3e-4, batch_size=64,
    existing_examples=examples,
)
save_model(model, "4x4x4_adi2", adi_history_2)

print("Post-ADI solve rate (beam_width=8, 20 trials)...")
post_rate = measure_solve_rate(model, n_trials=20, beam_width=8)
print(f"  Solve rate: {post_rate:.0%}")
print(f"\nImprovement: {pre_rate:.0%} → {post_rate:.0%}")