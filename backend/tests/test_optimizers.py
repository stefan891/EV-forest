"""End-to-end smoke tests for GA and NSGA-II.

These don't pin exact numerics (stochastic) but check load-bearing invariants:
the optimizer should *improve* over a no-cut baseline, and NSGA-II's reported
Pareto front should actually be non-dominated.
"""

from __future__ import annotations

import numpy as np

from ev_forest.fitness import FitnessConfig, evaluate
from ev_forest.forest import make_forest
from ev_forest.ga import run_ga
from ev_forest.nsga2 import _dominates, run_nsga2


def _small_dense_problem() -> FitnessConfig:
    forest = make_forest(10, 10, layout="dense")
    return FitnessConfig(
        forest_grid=forest.grid,
        ignition_strategy="random",
        ignition_samples=4,
        min_survival_rate=0.80,
        max_burn_rate=0.10,
        max_cut_rate=0.40,
    )


def test_no_cut_baseline_burns_dense_forest_fully():
    config = _small_dense_problem()
    no_cut = np.zeros_like(config.forest_grid)
    report = evaluate(no_cut, config)
    assert report.trees_burned == report.trees_original


def test_ga_improves_over_no_cut():
    config = _small_dense_problem()
    no_cut_report = evaluate(np.zeros_like(config.forest_grid), config)
    result = run_ga(
        config,
        population_size=40,
        max_generations=30,
        mutation_rate=0.02,
        seed=42,
    )
    assert result.best_report.trees_burned < no_cut_report.trees_burned
    assert result.best_report.scalar_fitness > no_cut_report.scalar_fitness
    assert len(result.history) == result.generations_run
    assert result.stopped_reason in {"fit_enough", "max_generations", "converged"}


def test_ga_stops_when_fit_enough():
    config = _small_dense_problem()
    # Lower thresholds so it's easy to clear them.
    config.min_survival_rate = 0.10
    config.max_burn_rate = 0.99
    config.max_cut_rate = 0.99
    result = run_ga(config, population_size=20, max_generations=50, seed=1)
    # With permissive thresholds, fit_enough should fire (no-cut already passes).
    assert result.stopped_reason == "fit_enough"


def test_nsga2_returns_nondominated_front():
    config = _small_dense_problem()
    result = run_nsga2(
        config,
        population_size=40,
        max_generations=20,
        mutation_rate=0.02,
        seed=7,
    )
    assert len(result.pareto_front) >= 1
    # No member of the reported front should dominate any other member.
    objs = [r.objectives for r in result.pareto_reports]
    for i, a in enumerate(objs):
        for j, b in enumerate(objs):
            if i == j:
                continue
            assert not _dominates(a, b), f"front not non-dominated: {a} dominates {b}"


def test_nsga2_front_spans_tradeoff():
    """The Pareto front should contain both a low-burn and a high-survival solution."""
    config = _small_dense_problem()
    result = run_nsga2(
        config,
        population_size=50,
        max_generations=30,
        mutation_rate=0.02,
        seed=11,
    )
    # objectives are (survived, -burned)
    min_burned = min(-r.objectives[1] for r in result.pareto_reports)
    max_survived = max(r.objectives[0] for r in result.pareto_reports)
    no_cut = evaluate(np.zeros_like(config.forest_grid), config)
    # NSGA-II should find at least one solution that burns less than no-cut.
    assert min_burned < no_cut.trees_burned
    # And at least one with positive survived count.
    assert max_survived > 0
