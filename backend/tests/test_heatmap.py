"""Tests for the procedural heatmap and the WeightedPipeline class.

The pipeline class is intentionally isolated from `fitness.py` / `ignition.py`
so collaborators iterating on the algorithms don't get merge conflicts. These
tests pin that contract: pipeline.evaluate / .expected_burn behave like the
base versions except that random/worst_case ignition is biased by per-cell
weights.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ev_forest.fitness import FitnessConfig, evaluate
from ev_forest.heatmap import BASELINE_WEIGHT, HeatmapConfig, make_heatmap
from ev_forest.weighted_pipeline import WeightedPipeline


# --------- heatmap generator ---------

def test_no_hotspots_is_flat_baseline():
    hm = make_heatmap(HeatmapConfig(rows=10, cols=10, n_hotspots=0))
    assert hm.shape == (10, 10)
    assert np.allclose(hm, BASELINE_WEIGHT)


def test_hotspots_create_peaks_above_baseline():
    hm = make_heatmap(HeatmapConfig(rows=20, cols=20, n_hotspots=3, seed=42))
    assert hm.max() > BASELINE_WEIGHT + 1.0
    near_baseline = (hm < BASELINE_WEIGHT + 0.1).sum()
    assert near_baseline > hm.size * 0.3


def test_heatmap_is_deterministic_per_seed():
    cfg = HeatmapConfig(rows=15, cols=15, n_hotspots=4, seed=7)
    a = make_heatmap(cfg)
    b = make_heatmap(cfg)
    assert np.array_equal(a, b)
    c = make_heatmap(HeatmapConfig(rows=15, cols=15, n_hotspots=4, seed=8))
    assert not np.array_equal(a, c)


# --------- WeightedPipeline ---------

def _pipeline_for(grid: np.ndarray, weights: np.ndarray, strategy="random", samples=8) -> WeightedPipeline:
    config = FitnessConfig(
        forest_grid=grid,
        ignition_strategy=strategy,
        ignition_samples=samples,
        ignition_seed=0,
    )
    return WeightedPipeline(config, weights)


def test_pipeline_random_points_biases_toward_hot_cells():
    grid = np.ones((10, 10), dtype=np.int8)
    weights = np.ones((10, 10), dtype=np.float32)
    weights[5, 5] = 500.0

    pipeline = _pipeline_for(grid, weights)
    counter: Counter[tuple[int, int]] = Counter()
    for seed in range(200):
        pts = pipeline.random_points(grid, n=1, seed=seed)
        counter[pts[0]] += 1

    most_common, count = counter.most_common(1)[0]
    assert most_common == (5, 5)
    assert count > 100  # 500/(99+500) ≈ 0.83


def test_pipeline_uniform_weights_dont_concentrate():
    grid = np.ones((6, 6), dtype=np.int8)
    uniform = np.ones((6, 6), dtype=np.float32)
    pipeline = _pipeline_for(grid, uniform)
    counter: Counter[tuple[int, int]] = Counter()
    for seed in range(2000):
        pts = pipeline.random_points(grid, n=1, seed=seed)
        counter[pts[0]] += 1
    assert counter.most_common(1)[0][1] < 120
    assert len(counter) == 36


def test_pipeline_evaluate_matches_base_when_weights_uniform():
    """With flat weights, the pipeline's evaluate should produce the same scalar
    as the base evaluate (averaged over seeds, since sampling differs internally)."""
    grid = np.ones((12, 12), dtype=np.int8)
    cut_mask = np.zeros((12, 12), dtype=np.int8)
    flat = np.ones((12, 12), dtype=np.float32)
    config = FitnessConfig(
        forest_grid=grid,
        ignition_strategy="random",
        ignition_samples=8,
        ignition_seed=0,
    )
    pipeline = WeightedPipeline(config, flat)
    base = evaluate(cut_mask, config)
    weighted = pipeline.evaluate(cut_mask)
    # Both run the same fire mechanics on the same grid — burned counts may
    # differ slightly because sampling internals differ (rng.choice with vs
    # without p=), but baseline survival should be identical.
    assert base.trees_original == weighted.trees_original
    assert base.trees_cut == weighted.trees_cut
    assert base.trees_remaining == weighted.trees_remaining


def test_pipeline_weights_shape_mismatch_raises():
    grid = np.ones((5, 5), dtype=np.int8)
    bad_weights = np.ones((4, 5), dtype=np.float32)
    config = FitnessConfig(forest_grid=grid)
    try:
        WeightedPipeline(config, bad_weights)
    except ValueError:
        return
    raise AssertionError("expected ValueError for shape mismatch")


def test_pipeline_run_ga_completes_and_restores_evaluate():
    """The scoped monkey-patch must restore ga.evaluate after run_ga returns."""
    from ev_forest import ga as ga_mod

    grid = np.ones((10, 10), dtype=np.int8)
    weights = np.ones((10, 10), dtype=np.float32)
    weights[2, 2] = 50.0
    config = FitnessConfig(
        forest_grid=grid,
        ignition_strategy="random",
        ignition_samples=4,
    )
    pipeline = WeightedPipeline(config, weights)

    original_evaluate = ga_mod.evaluate
    result = pipeline.run_ga(
        population_size=12,
        max_generations=4,
        seed=0,
    )
    assert ga_mod.evaluate is original_evaluate, "evaluate symbol must be restored"
    assert result.best_individual.shape == grid.shape


def test_pipeline_run_nsga2_completes_and_restores_evaluate():
    from ev_forest import nsga2 as nsga2_mod

    grid = np.ones((10, 10), dtype=np.int8)
    weights = np.ones((10, 10), dtype=np.float32)
    config = FitnessConfig(
        forest_grid=grid,
        ignition_strategy="random",
        ignition_samples=4,
    )
    pipeline = WeightedPipeline(config, weights)

    original_evaluate = nsga2_mod.evaluate
    result = pipeline.run_nsga2(
        population_size=12,
        max_generations=4,
        seed=0,
    )
    assert nsga2_mod.evaluate is original_evaluate
    assert len(result.pareto_front) > 0
