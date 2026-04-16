import modal

app = modal.App("polycube-nn-solver")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
    .add_local_dir(".", remote_path="/root/polycube", ignore=[".venv", "__pycache__", "*.pyc"])
)


# use T4 GPU for fast inference on neural network
@app.function(image=image, timeout=120, gpu="T4") # can use other GPUs too. https://modal.com/pricing
def solve_4x4x4(beam_width=64, timeout=60.0):
    import sys
    import time
    sys.path.insert(0, "/root/polycube")

    from phase2.data_generator import enumerate_polycubes, generate_puzzle_instances
    from phase2.nn_solver import solve_with_nn

    print("Generating 4x4x4 puzzle instance...")
    catalog = enumerate_polycubes(max_size=5)
    instances = generate_puzzle_instances(
        num_instances=1,
        grid_size=4,
        polycube_catalog=catalog,
        min_piece_size=3,
        max_piece_size=5,
        dlx_timeout=30.0,
        verbose=False,
    )
    pieces = instances[0]["pieces"]
    print(f"Puzzle: {len(pieces)} pieces, grid=4x4x4")

    print(f"Running NN solver (model=4x4x4_adi2, beam_width={beam_width})...")
    t0 = time.time()
    solution = solve_with_nn(
        pieces,
        grid_size=4,
        model_name="4x4x4_adi2",
        beam_width=beam_width,
        timeout=timeout,
        device="cuda",
    )
    elapsed = time.time() - t0

    if solution:
        print(f"Solution found in {elapsed:.3f}s!")
    else:
        print(f"No solution found after {elapsed:.3f}s.")

    return elapsed, solution is not None


@app.local_entrypoint()
def main():
    elapsed, solved = solve_4x4x4.remote()
    status = "solved" if solved else "failed"
    print(f"\n{status} in {elapsed:.3f}s")
