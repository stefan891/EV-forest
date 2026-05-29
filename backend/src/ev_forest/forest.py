"""Forest grid construction and serialization.

A forest is a 2D numpy array of dtype int8 where 0 = empty and 1 = tree.
The Forest class wraps the array with the metadata that describes how it was
generated (so a run can be reproduced from a seed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

EMPTY: int = 0
TREE: int = 1

Layout = Literal["baseline", "dense", "random", "checkerboard"]


@dataclass
class Forest:
    grid: np.ndarray
    layout: Layout
    seed: int
    density: float = 1.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape  # type: ignore[return-value]

    @property
    def tree_count(self) -> int:
        return int(self.grid.sum())

    def with_cut(self, cut_mask: np.ndarray) -> np.ndarray:
        """Return the grid after applying a cut mask (1 = cut)."""
        if cut_mask.shape != self.grid.shape:
            raise ValueError(
                f"cut_mask shape {cut_mask.shape} != forest shape {self.grid.shape}"
            )
        return (self.grid & (1 - cut_mask.astype(np.int8))).astype(np.int8)


def make_forest(
    rows: int,
    cols: int,
    layout: Layout = "dense",
    density: float = 1.0,
    seed: int = 0,
) -> Forest:
    """Generate a forest grid.

    - dense: every cell is a tree (worst case for fire spread).
    - baseline: trees every 2 cells in each direction; each tree has a 1-cell
      empty ring. Fire-safe by construction — used as a reference floor.
    - random: each cell is a tree with probability `density`.
    - checkerboard: alternating tree/empty pattern.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be in [0, 1]")

    grid = np.zeros((rows, cols), dtype=np.int8)

    if layout == "dense":
        grid[:] = TREE
    elif layout == "baseline":
        grid[::2, ::2] = TREE
    elif layout == "checkerboard":
        grid[(np.add.outer(np.arange(rows), np.arange(cols)) % 2) == 0] = TREE
    elif layout == "random":
        rng = np.random.default_rng(seed)
        grid = (rng.random((rows, cols)) < density).astype(np.int8)
    else:
        raise ValueError(f"unknown layout: {layout}")

    return Forest(grid=grid, layout=layout, seed=seed, density=density)


def serialize(grid: np.ndarray) -> list[list[int]]:
    """Convert a grid to a JSON-serializable nested list."""
    return grid.astype(int).tolist()
