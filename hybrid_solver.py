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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1.solver import solve as dlx_solve, cube_root_int
from phase2.nn_solver import nn_solve
from phase2.train import load_model


def hybrid_solve(pieces, grid_size=None, model_name="soma_3x3x3",
                 beam_width=64, timeout_nn=30, timeout_dlx=120,
                 device='cpu', verbose=True):
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

    Returns:
        dict: {
            'solution': dict mapping piece_idx -> frozenset of cells, or None,
            'method': 'nn' | 'dlx' | None,
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

    # ── Try NN solver first ──
    nn_solution = None
    nn_time = 0.0

    if model_name is not None:
        try:
            if verbose:
                print(f"\n[1] NN beam search (beam_width={beam_width}, "
                      f"timeout={timeout_nn}s)...")

            model, _, metadata = load_model(model_name, device=device)
            max_pieces = model.in_channels - 1

            t_nn = time.time()
            nn_solution = nn_solve(
                pieces, grid_size, model,
                max_pieces=max_pieces, beam_width=beam_width,
                timeout=timeout_nn, device=device,
            )
            nn_time = time.time() - t_nn

            if nn_solution is not None:
                if verbose:
                    print(f"    NN solved in {nn_time:.3f}s!")
                return {
                    'solution': nn_solution,
                    'method': 'nn',
                    'time': nn_time,
                }
            else:
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
    dlx_solutions = dlx_solve(pieces, grid_size=grid_size, find_all=False)
    dlx_time = time.time() - t_dlx

    if dlx_solutions:
        if verbose:
            print(f"    DLX solved in {dlx_time:.3f}s")
        return {
            'solution': dlx_solutions[0],
            'method': 'dlx',
            'time': nn_time + dlx_time,
        }
    else:
        if verbose:
            print(f"    DLX found no solution ({dlx_time:.3f}s)")
        return {
            'solution': None,
            'method': None,
            'time': nn_time + dlx_time,
        }


def compare_solvers(pieces, grid_size=None, model_name="soma_3x3x3",
                    beam_width=64, device='cpu', verbose=True):
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
