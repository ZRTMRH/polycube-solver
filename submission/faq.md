# Frequently Asked Questions

**STA 561 Final Project --- Professor Cubazoid's 3D Tetris**
*Team: ZRTMRH, Nicholas Popescu, Isabelle Dorage, Rally Lin*

---

### Q1: How is this different from brute-force search?

A pure brute-force approach would enumerate every possible placement of every piece in every orientation and check all combinations. For a 3x3x3 cube with 7 pieces, this is already on the order of millions of states. For a 10x10x10 cube with ~200 pieces, the combinatorial space is astronomically large --- far beyond what any computer could enumerate.

Our solver avoids brute force in three ways. First, the Dancing Links (DLX) exact cover formulation reduces the problem to a structured matrix, and Algorithm X prunes vast swaths of the search tree by choosing the most constrained column first (analogous to solving the hardest part of a jigsaw puzzle before the easy parts). Second, the neural network provides a learned heuristic that predicts which partial assemblies are likely completable, allowing beam search to focus on the most promising paths rather than exploring blindly. Third, block decomposition breaks a problem with, say, 1,000 cells into eight independent 125-cell sub-problems, reducing complexity multiplicatively rather than additively.

### Q2: Why not just use the neural network by itself?

The neural network is fast but not guaranteed to find a solution even when one exists. Beam search is an incomplete algorithm: it keeps only the top-K most promising states at each depth, so it can miss solutions that require exploring less obvious paths. In our testing, the NN beam search alone solves about 75--85% of 3x3x3 instances (with a narrow beam). It is useful as a quick first attempt but not reliable enough on its own.

More importantly, the NN was only trained on 3x3x3 data. While we implemented scale-invariant architecture features (global average pooling, convolutional policy heads) that allow the model to accept any grid size, we have not yet trained on larger instances. The NN's heuristic quality degrades outside its training distribution.

The DLX exact solver, by contrast, is guaranteed correct --- if a solution exists, it will find it (given enough time). The hybrid approach gets the best of both: try the fast NN first, fall back to the reliable DLX.

### Q3: How do you handle unsolvable inputs?

The solver employs several checks:

1. **Volume validation.** If the total volume of all pieces does not equal a perfect cube (N^3 for some integer N), the input is immediately rejected.

2. **Exhaustive search.** For small grids, the DLX solver explores the entire search tree. If it completes without finding a solution, the input is provably unsolvable.

3. **Timeout-based rejection.** For large grids where exhaustive search is infeasible, the solver attempts all available strategies (block decomposition, NN search, DLX) within configurable time limits. If none succeeds, it returns null. This is technically incomplete --- we cannot guarantee the input is unsolvable, only that we could not solve it within the time budget.

4. **Oversized piece detection.** Before attempting to solve, we check whether any piece has a bounding box that exceeds the grid dimensions. Such pieces cannot fit regardless of orientation, so the input is rejected immediately.

In our testing with 323 fault cases (structurally unsolvable inputs), the solver never produced an incorrect solution. The most common outcomes were immediate rejection (for oversized pieces or volume mismatches) and timeout (for subtly corrupted inputs where the solver correctly fails to find a nonexistent solution).

### Q4: How does the block decomposition work? Why split into sub-cubes?

The core insight is that the DLX solver handles grids up to 9x9x9 efficiently (seconds to minutes), but becomes impractically slow for larger grids because the search tree grows exponentially with volume. Instead of solving one huge problem, we decompose it into many small ones.

For a grid of size N >= 10, we split each axis independently into parts in the range 5--9 and partition accordingly. For example:

- **N=10**: Split as 5+5 on each axis, creating 8 sub-cubes of 5x5x5 each.
- **N=14**: Split as 7+7, creating 8 sub-cubes of 7x7x7 each.
- **N=11**: Split as 6+5, creating rectangular sub-boxes (6x6x6, 6x6x5, 6x5x5, 5x5x5).
- **N=17**: Split as 6+6+5, creating 27 sub-boxes of various rectangular shapes.

The solver then allocates pieces to sub-boxes (matching total volume) and solves each sub-box independently with DLX. If a particular allocation fails, it retries with a different random assignment. This approach transforms an intractable N^3-cell problem into a collection of tractable sub-problems, each with at most ~729 cells.

The rectangular sub-box extension was a critical breakthrough. Many grid sizes (11, 13, 17, 19, 23) are prime or have no useful divisors, making pure cubic decomposition impossible. The rectangular approach handles all integers by splitting each axis independently.

A natural concern is that decomposition might miss solutions where pieces straddle sub-box boundaries. In practice this never happens, because pieces have at most 5 cells while sub-boxes have minimum dimension 5 --- so pieces never *need* to cross boundaries. Our test generator produces random decompositions that do not respect block boundaries, yet across 380+ solvable cases (N=10 to N=24), the solver always found a valid re-allocation. Small piece sizes provide enough combinatorial flexibility that some valid allocation to sub-boxes always exists.

### Q5: How did you validate that your solutions are actually correct?

We implemented a rigorous independent verification system that checks four properties of every returned solution:

1. **Completeness**: Every cell in the N x N x N grid is covered by exactly one piece.
2. **Piece usage**: Every input piece appears exactly once in the solution.
3. **Shape matching**: Each placed piece, when translated to the origin and compared against all 24 rotations of the original piece, matches at least one rotation.
4. **Bounds checking**: No piece extends outside the grid boundaries.

This verifier runs after every solve, in every test. Across all 1,800+ test cases we have run, zero verification failures have occurred.

We also tested with five distinct categories of intentionally unsolvable inputs --- oversized pieces, all-rod pieces, 40% piece swaps, shape duplication, and fully random piece sets --- to confirm the solver never hallucinates a solution for an impossible input.

### Q6: What is Autodidactic Iteration and why does it help?

Autodidactic Iteration (ADI), from the DeepCube paper by Agostinelli et al. (2019), is a self-improvement loop for the neural network:

1. Generate puzzle instances and attempt to solve them with the current model using a deliberately narrow beam search (width 8 instead of 64).
2. For puzzles the model solves successfully, label all visited states as "solvable" (positive examples).
3. For puzzles where the model fails, label the visited states as "unsolvable" (negative examples) --- these are states the model incorrectly scored as promising.
4. Retrain the model on the combined original and new data.

The narrow beam is intentional: it makes the model fail more often, generating informative negative examples. After two ADI rounds on 3x3x3 Soma cubes, the narrow-beam solve rate improved from approximately 55% to 80%. The model learns to avoid dead-end configurations it previously misjudged.

### Q7: How did AI (Claude) assist in developing this project?

AI assistance was integral to our development process, used as a collaborator rather than a replacement for understanding. Specific contributions include:

- **Algorithm implementation**: Claude helped implement the Dancing Links data structure and Algorithm X, translating Knuth's theoretical description into working Python code with proper pointer management.
- **Architecture iteration**: We explored multiple solver strategies (slab planners, rod planners, pair planners) with rapid prototyping before finding that block decomposition was the right approach for scaling. Claude enabled us to test and discard ideas in hours rather than days.
- **Debugging and optimization**: When the solver stalled on specific grid sizes (e.g., N=16 getting stuck on 8x8x8 blocks), Claude helped diagnose the bottleneck and tune parameters (reducing maximum block volume from 512 to 343 cells).
- **Human-guided algorithmic insight**: Several critical improvements came from human observation of AI-generated profiling data. For example, Claude's profiling revealed that the puzzle generator spent 94% of its time on connectivity checks, with 97% of candidate cell removals failing. By examining *why* they failed, we observed that cells in the "middle" of the remaining structure almost always cause disconnections when removed. This led to sorting surface cells by ascending neighbor count (fewest neighbors = most peripheral = least likely to disconnect), yielding a 7x speedup. Similarly, the generator's piece-construction strategy was redesigned based on human reasoning: instead of enumerating all possible pieces across the entire surface, we first select a single surface cube and then enumerate only pieces containing that cube --- a change that made the generation process both faster and more controllable.
- **Test infrastructure**: The comprehensive test suite --- 1,800+ cases with five fault categories, per-block logging, wall-clock timeouts, and independent verification --- was developed iteratively with AI assistance.
- **Documentation**: This submission itself was drafted with AI assistance, though all technical content reflects our actual implementation and results.

The key lesson: AI dramatically accelerates iteration speed. Ideas that would take a day to implement and test could be explored in an hour. This allowed us to attempt and evaluate far more approaches than would otherwise have been feasible, ultimately arriving at a more sophisticated and robust solution.

### Q8: What are the main limitations of your solver?

1. **Test generation bottleneck.** Our constructive puzzle generator (which builds valid decompositions by randomly carving pieces from a cube) becomes very slow for N >= 18, taking up to 10 minutes per instance. This limits our ability to generate test cases at large scales, though the solver itself handles them fine.

2. **No unsolvability proof for large grids.** For grids larger than about 9x9x9, the DLX solver cannot exhaustively search the entire space in reasonable time. This means we cannot provably certify that an input is unsolvable --- we can only report failure to find a solution within the time budget.

3. **Neural network limited to small grids.** The trained model only covers 3x3x3. While the architecture supports arbitrary sizes, training on larger grids requires substantially more compute and data. The NN currently contributes little to solving grids larger than 5x5x5.

4. **Block allocation is randomized.** The piece-to-block allocation in decomposition uses random sampling with retries. For some piece distributions, many retries may be needed before finding a feasible allocation. A smarter allocation algorithm (e.g., constraint-based or learned) could improve reliability.

5. **Piece size assumption.** The solver assumes all pieces have 3, 4, or 5 cells, as specified in the project requirements. Extending to arbitrary piece sizes would require adjusting the block decomposition logic and retraining the neural network.

### Q9: How confident are you in meeting the A+ threshold (15/20 test cases)?

Very confident for the specified problem scope (pieces of sizes 3--5 assembling into perfect cubes). Our solver achieves 100% solve rate across all tested grid sizes from 3 to 24, with independent verification of every solution. The main risk factors for the professor's test cases are:

- **Grid sizes we haven't tested.** We have tested N=3 through N=24 with hundreds of seeds. Unless the professor uses N >= 25, we expect our solver to handle any grid size within this range.
- **Time limits.** Our solver takes up to ~5 minutes for the largest grids (N=18). If the professor imposes a strict per-case time limit shorter than this, some large cases might time out.
- **Edge cases.** Unusual piece distributions (e.g., all pieces the same shape, or many size-3 pieces) could affect block allocation. We tested with randomized piece distributions and five fault categories to cover these scenarios.

Based on our A+ fixture benchmark (150 stratified cases, N=3 to N=12), the estimated probability of scoring at least 15/20 on a random 20-case test set is above 99%.
