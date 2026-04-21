import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from phase1.test_cases import SOMA_PIECES
from phase1.visualization import PIECE_COLORS

piece = SOMA_PIECES[0]

filled = np.zeros((4, 4, 4), dtype=bool)
for x, y, z in piece:
    filled[x, y, z] = True

color = PIECE_COLORS[0]
facecolors = np.where(filled[..., np.newaxis],
                      np.array(mcolors.to_rgba(color, alpha=0.85)),
                      np.array([0, 0, 0, 0]))
edgecolors = np.where(filled[..., np.newaxis],
                      np.array([0, 0, 0, 0.3]),
                      np.array([0, 0, 0, 0]))

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.voxels(filled, facecolors=facecolors, edgecolors=edgecolors)
ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.set_zlim(0, 4)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_zlabel('')
plt.show()
