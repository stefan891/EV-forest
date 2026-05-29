"""Tests pinning down the load-bearing fire-spread invariants from CLAUDE.md."""

from __future__ import annotations

import numpy as np
import pytest

from ev_forest.forest import TREE, make_forest
from ev_forest.simulator import simulate_fire, simulate_with_cut


def test_baseline_layout_is_fire_safe():
    """Per CLAUDE.md: every tree surrounded by a 1-cell empty ring => no spread."""
    forest = make_forest(11, 11, layout="baseline")
    # Ignite the center tree (which exists in baseline at (0,0), (0,2), ..., so (10,10)).
    # Center of 11x11 is (5,5) — odd index, not a tree in baseline. Pick (4,4) instead.
    assert forest.grid[4, 4] == TREE
    result = simulate_fire(forest.grid, [(4, 4)])
    assert result.burned == 1, "isolated tree should not spread"
    assert result.steps == 0


def test_dense_forest_burns_completely():
    forest = make_forest(8, 8, layout="dense")
    result = simulate_fire(forest.grid, [(0, 0)])
    assert result.burned == 64
    assert result.survived == 0


def test_diagonal_spread_works():
    """8-neighbor adjacency: a diagonal contact lets fire jump."""
    grid = np.array(
        [[1, 0],
         [0, 1]],
        dtype=np.int8,
    )
    result = simulate_fire(grid, [(0, 0)])
    assert result.burned == 2, "diagonals must spread"


def test_one_pixel_gap_blocks_spread():
    """Two trees separated by one empty cell must not co-burn."""
    grid = np.array(
        [[1, 0, 1]],
        dtype=np.int8,
    )
    result = simulate_fire(grid, [(0, 0)])
    assert result.burned == 1, "1-cell gap must block spread"


def test_two_pixel_gap_also_blocks():
    grid = np.array(
        [[1, 0, 0, 1]],
        dtype=np.int8,
    )
    result = simulate_fire(grid, [(0, 0)])
    assert result.burned == 1


def test_adjacent_horizontal_burns():
    grid = np.array(
        [[1, 1, 1, 1]],
        dtype=np.int8,
    )
    result = simulate_fire(grid, [(0, 0)])
    assert result.burned == 4
    assert result.steps == 3  # spreads one step at a time


def test_ignition_on_empty_cell_does_nothing():
    grid = np.array(
        [[0, 1],
         [1, 0]],
        dtype=np.int8,
    )
    result = simulate_fire(grid, [(0, 0)])
    assert result.burned == 0


def test_out_of_bounds_ignition_is_ignored():
    grid = np.ones((3, 3), dtype=np.int8)
    result = simulate_fire(grid, [(-1, 0), (3, 3), (10, 10)])
    assert result.burned == 0


def test_cut_mask_removes_trees_before_burn():
    grid = np.ones((3, 3), dtype=np.int8)
    cut = np.zeros((3, 3), dtype=np.int8)
    cut[1, :] = 1  # cut the entire middle row -> creates a 1-cell gap
    result = simulate_with_cut(grid, cut, [(0, 0)])
    # Top row (3 trees) should burn; bottom row (3 trees) should survive.
    assert result.burned == 3
    assert result.survived == 3


def test_cut_mask_shape_mismatch_raises():
    grid = np.ones((3, 3), dtype=np.int8)
    bad_cut = np.zeros((4, 4), dtype=np.int8)
    with pytest.raises(ValueError):
        simulate_with_cut(grid, bad_cut, [(0, 0)])


def test_burned_per_step_records_animation():
    grid = np.ones((1, 5), dtype=np.int8)
    result = simulate_fire(grid, [(0, 0)])
    # Step 0 = ignition; steps 1..4 = spread one cell at a time.
    assert [len(s) for s in result.burned_per_step] == [1, 1, 1, 1, 1]


def test_total_trees_excludes_empty():
    grid = np.array(
        [[1, 0, 1],
         [0, 0, 0],
         [1, 0, 1]],
        dtype=np.int8,
    )
    result = simulate_fire(grid, [])
    assert result.total_trees == 4
    assert result.burned == 0
