"""Ignition strategies — how the fire starts.

This is a modeling decision called out in CLAUDE.md: the burned-tree objective
depends on what gets ignited. We expose three strategies; the optimizer picks
one and the fitness function uses it consistently across all individuals.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .forest import TREE
from .simulator import SimulationResult, simulate_fire

Strategy = Literal["fixed", "random", "worst_case"]


def fixed_point(rows: int, cols: int, point: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    """Single deterministic ignition cell (center by default)."""
    if point is None:
        return [(rows // 2, cols // 2)]
    return [point]


def random_points(
    grid: np.ndarray,
    n: int = 1,
    seed: int = 0,
) -> list[tuple[int, int]]:
    """Sample `n` random tree cells as ignition points."""
    rng = np.random.default_rng(seed)
    tree_positions = np.argwhere(grid == TREE)
    if len(tree_positions) == 0:
        return []
    idx = rng.choice(len(tree_positions), size=min(n, len(tree_positions)), replace=False)
    return [(int(r), int(c)) for r, c in tree_positions[idx]]


def expected_burn(
    grid: np.ndarray,
    strategy: Strategy = "fixed",
    samples: int = 8,
    seed: int = 0,
    fixed_at: tuple[int, int] | None = None,
) -> SimulationResult:
    """Return a representative SimulationResult for the chosen strategy.

    - fixed: ignite at `fixed_at` (or grid center).
    - random: average burn count over `samples` random ignitions; the returned
      `final_grid` is from the last sample (animation may not be representative
      under random — use single ignition for animation).
    - worst_case: try `samples` random starting points, keep the worst.
    """
    rows, cols = grid.shape
    if strategy == "fixed":
        return simulate_fire(grid, fixed_point(rows, cols, fixed_at))

    rng = np.random.default_rng(seed)
    tree_positions = np.argwhere(grid == TREE)
    if len(tree_positions) == 0:
        return simulate_fire(grid, [])

    n_samples = min(samples, len(tree_positions))
    sample_idx = rng.choice(len(tree_positions), size=n_samples, replace=False)
    results = [
        simulate_fire(grid, [(int(r), int(c))])
        for r, c in tree_positions[sample_idx]
    ]

    if strategy == "worst_case":
        return max(results, key=lambda r: r.burned)

    # "random": average burn across samples, but return last result's grid for shape.
    avg_burned = int(round(sum(r.burned for r in results) / len(results)))
    last = results[-1]
    return SimulationResult(
        total_trees=last.total_trees,
        burned=avg_burned,
        survived=last.total_trees - avg_burned,
        steps=last.steps,
        burned_per_step=last.burned_per_step,
        final_grid=last.final_grid,
    )
