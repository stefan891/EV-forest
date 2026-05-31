"""Ignition strategies — how the fire starts.

This is a modeling decision: the burned-tree objective depends on what gets
ignited. We expose two strategies; the optimizer picks one and the fitness
function uses it consistently across all individuals.

- random: Sample n random tree cells, pick the one burning the most.
- worst_case: BFS exhaustive search to find the true worst-case ignition point.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .forest import TREE
from .simulator import SimulationResult, simulate_fire

Strategy = Literal["random", "worst_case", "heatmap"]
# def fixed_point(rows: int, cols: int, point: tuple[int, int] | None = None) -> list[tuple[int, int]]:
#     """Single deterministic ignition cell (center by default)."""
#     if point is None:
#         return [(rows // 2, cols // 2)]
#     return [point]

def expected_burn(
    grid: np.ndarray,
    strategy: Strategy = "random",
    samples: int = 8,
    seed: int = 0,
    heatmap: np.ndarray | None = None,
) -> SimulationResult:
    """Return a representative SimulationResult for the chosen strategy.

    - random: Sample `samples` random tree cells, return the one that burns most.
      The returned `final_grid` is from that worst sample (animation shows this burn).
    - worst_case: BFS exhaustive search across all trees to find the true worst-case
      ignition point (uses memoization to avoid redundant simulations).
    - heatmap: Sample `samples` random tree cells weighted by the provided heatmap
    """
    tree_positions = np.argwhere(grid == TREE)
    if len(tree_positions) == 0:
        return simulate_fire(grid, [])

    if strategy == "random":
        # Sample n random trees and pick the worst burn
        rng = np.random.default_rng(seed)
        n_samples = min(samples, len(tree_positions))
        sample_idx = rng.choice(len(tree_positions), size=n_samples, replace=False)
        results = [
            simulate_fire(grid, [(int(r), int(c))])
            for r, c in tree_positions[sample_idx]
        ]
        return max(results, key=lambda r: r.burned)

    if strategy == "heatmap":
        # Use the provided heatmap to weight sampled ignition points. If no
        # heatmap is provided, fall back to uniform random sampling.
        rng = np.random.default_rng(seed)
        weights = None
        if heatmap is not None:
            # Collect weights for tree positions only
            w = []
            for r, c in tree_positions:
                val = float(heatmap[int(r)][int(c)])
                w.append(max(0.0, val))
            total = sum(w)
            if total > 0:
                # Normalize weights
                weights = np.array(w) / total

        n_samples = min(samples, len(tree_positions))
        if weights is None:
            # Uniform sampling
            sample_idx = rng.choice(len(tree_positions), size=n_samples, replace=False)
        else:
            # Weighted sampling without replacement: use choice with p=weights
            sample_idx = rng.choice(len(tree_positions), size=n_samples, replace=False, p=weights)

        results = [
            simulate_fire(grid, [(int(r), int(c))])
            for r, c in tree_positions[sample_idx]
        ]
        return max(results, key=lambda r: r.burned)

    # "worst_case": exhaustive BFS search with memoization
    memo = {}

    def simulate_from(r: int, c: int) -> int:
        """Simulate fire from (r,c) and return burn count (memoized)."""
        key = (r, c)
        if key not in memo:
            result = simulate_fire(grid, [(r, c)])
            memo[key] = result.burned
        return memo[key]

    # Test every tree position, find the one that burns most
    max_burned = -1
    worst_point = (0, 0)
    worst_result = None

    for r, c in tree_positions:
        burned = simulate_from(int(r), int(c))
        if burned > max_burned:
            max_burned = burned
            worst_point = (int(r), int(c))
            worst_result = simulate_fire(grid, [worst_point])

    return worst_result if worst_result else simulate_fire(grid, [worst_point])
