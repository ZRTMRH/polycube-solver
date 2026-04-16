"""Quick test of the slab_layered solver on normalized flat pieces."""
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grading_harness import build_mixed_constructive_case
from block_planner_v2 import solve_slab_layered, detect_slab_axis, _precompute_2d_shapes

# Test at N=7, 9, 12 with relative_pieces=True
for N in [7, 9, 12]:
    print(f"\n{'='*50}")
    print(f"Testing slab_layered at N={N} (relative_pieces=True)")
    print(f"{'='*50}")
    sys.stdout.flush()

    pieces = build_mixed_constructive_case(N, seed=561 + N, relative_pieces=True)
    print(f"  {len(pieces)} pieces, {sum(len(p) for p in pieces)} cells")
    sys.stdout.flush()

    axis = detect_slab_axis(pieces)
    print(f"  slab_axis={axis}")
    sys.stdout.flush()

    # Check piece size distribution
    from collections import Counter
    sizes = Counter(len(p) for p in pieces)
    print(f"  piece sizes: {dict(sizes)}")
    sys.stdout.flush()

    t0 = time.time()
    solution, diag = solve_slab_layered(
        pieces, N,
        slab_timeout=10.0,
        total_timeout=120.0,
        max_retries=100,
    )
    elapsed = time.time() - t0

    if solution is not None:
        # Verify solution
        all_cells = set()
        for pidx, cells in solution.items():
            all_cells |= cells
        expected = {(x,y,z) for x in range(N) for y in range(N) for z in range(N)}
        valid = all_cells == expected and len(all_cells) == N**3
        print(f"  PASS ({elapsed:.1f}s) valid={valid} retries={diag.get('retries',0)}")
    else:
        print(f"  FAIL ({elapsed:.1f}s) reason={diag.get('reason')} retries={diag.get('retries',0)}")
    sys.stdout.flush()
