"""Deterministic fire-spread simulator.

Fire propagates from a set of ignition cells via 8-neighbor adjacency through
tree cells. A single empty cell blocks spread — fire cannot jump a gap. This is
the source of truth that every optimizer calls into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .forest import EMPTY, TREE

# 8 directions, including diagonals. Order doesn't matter — BFS visits all.
NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
)


@dataclass
class SimulationResult:
    """Outcome of one fire on a given grid."""

    total_trees: int          # tree count before ignition
    burned: int               # cells that ended up burning
    survived: int             # tree cells that didn't burn
    steps: int                # number of BFS rounds the fire ran
    burned_per_step: list[list[tuple[int, int]]]  # for animation playback
    final_grid: np.ndarray    # 0=empty, 1=tree (survived), 2=burned

    @property
    def burn_fraction(self) -> float:
        return self.burned / self.total_trees if self.total_trees else 0.0


def simulate_fire(
    grid: np.ndarray,
    ignition: Iterable[tuple[int, int]],
) -> SimulationResult:
    """Run a deterministic fire on `grid` starting from `ignition` cells.

    Ignition cells that are empty are silently skipped — they cannot burn.
    Returns a SimulationResult capturing what burned and the per-step expansion
    (useful for replay in the frontend).
    """
    rows, cols = grid.shape
    total_trees = int((grid == TREE).sum())

    # state: 0=empty, 1=tree (alive), 2=burned
    state = grid.astype(np.int8, copy=True)

    # Seed the fire on tree cells only.
    frontier: list[tuple[int, int]] = []
    for r, c in ignition:
        if 0 <= r < rows and 0 <= c < cols and state[r, c] == TREE:
            state[r, c] = 2
            frontier.append((r, c))

    burned_per_step: list[list[tuple[int, int]]] = []
    if frontier:
        burned_per_step.append(list(frontier))

    steps = 0
    while frontier:
        next_frontier: list[tuple[int, int]] = []
        for r, c in frontier:
            for dr, dc in NEIGHBOR_OFFSETS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and state[nr, nc] == TREE:
                    state[nr, nc] = 2
                    next_frontier.append((nr, nc))
        if not next_frontier:
            break
        burned_per_step.append(next_frontier)
        frontier = next_frontier
        steps += 1

    burned = int((state == 2).sum())
    survived = total_trees - burned
    return SimulationResult(
        total_trees=total_trees,
        burned=burned,
        survived=survived,
        steps=steps,
        burned_per_step=burned_per_step,
        final_grid=state,
    )


def simulate_with_cut(
    forest_grid: np.ndarray,
    cut_mask: np.ndarray,
    ignition: Iterable[tuple[int, int]],
) -> SimulationResult:
    """Apply a cut mask, then simulate fire on the resulting forest."""
    if cut_mask.shape != forest_grid.shape:
        raise ValueError("cut_mask shape mismatch")
    remaining = (forest_grid & (1 - cut_mask.astype(np.int8))).astype(np.int8)
    return simulate_fire(remaining, ignition)
