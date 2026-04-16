
from phase1.polycube import get_orientations, get_all_placements
from phase1.solver import solve
from phase1.visualization import plot_solution, animate_solution, plot_pieces
import matplotlib.pyplot as plt
from phase1.test_cases import (
    MONOMINO, DOMINO, L_TRICUBE, SOMA_PIECES,
    verify_solution
)
# The L-tricube has how many distinct orientations?
orientations = get_orientations(L_TRICUBE)
print(f"L-tricube: {len(L_TRICUBE)} cubes, {len(orientations)} unique orientations")
for i, o in enumerate(orientations):
    print(f"  Orientation {i}: {sorted(o)}")
pieces_2x2 = [DOMINO] * 4
solutions = solve(pieces_2x2)

if solutions:
    sol = solutions[0]
    print("\nSolution:")
    for pidx, cells in sorted(sol.items()):
        print(f"  Piece {pidx}: {sorted(cells)}")
    assert verify_solution(sol, 2), "Solution verification failed!"
    print("\nVerified: all cells covered exactly once.")
fig, ax = plot_solution(sol, 2, title="2×2×2 Cube — 4 Dominoes")
plt.show()
# Visualize the 7 Soma pieces
fig = plot_pieces(SOMA_PIECES, title="The 7 Soma Cube Pieces")
plt.show()
# Solve the Soma cube
soma_solutions = solve(SOMA_PIECES)

if soma_solutions:
    soma_sol = soma_solutions[0]
    print("\nSolution:")
    for pidx, cells in sorted(soma_sol.items()):
        print(f"  Piece {pidx}: {sorted(cells)}")
    assert verify_solution(soma_sol, 3), "Solution verification failed!"
    print("\nVerified: perfect 3×3×3 cube!")
# Static visualization of the Soma solution
fig, ax = plot_solution(soma_sol, 3, title="Soma Cube Solution")
plt.show()
# Animated assembly — pieces placed one at a time
anim = animate_solution(soma_sol, 3, title="Soma Cube Assembly", interval=1000)
plt.show()
# 3 dominoes = 6 cubes — not a perfect cube
impossible = solve([DOMINO] * 3)
print(f"Solutions found: {len(impossible)}")
import time

t0 = time.time()
all_soma = solve(SOMA_PIECES, find_all=True)
elapsed = time.time() - t0

print(f"\nTotal solutions: {len(all_soma)}")
print(f"Unique (mod 48 symmetries): {len(all_soma) // 48}")
print(f"Time: {elapsed:.2f}s")
# Save the Soma animation as a GIF
anim = animate_solution(soma_sol, 3, title="Soma Cube Assembly",
                        interval=1000, save_path="soma_assembly.gif")