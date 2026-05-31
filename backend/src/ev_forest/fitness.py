"""Fitness evaluation for cut-mask chromosomes.

Two views of the same simulation, both consumed by optimizers:

- Single-objective scalar fitness for the plain GA: a weighted combination of
  survival, burn, and cut counts.
- Multi-objective tuple `(survived, -burned)` for NSGA-II: both to maximize,
  expressed in the convention `(maximize, maximize)` so the optimizer can sort
  without sign confusion.

The `is_fit_enough` and `is_population_fit_enough` helpers answer the question
the user asked directly: given thresholds, is an individual / the population
good enough to stop searching?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .ignition import Strategy, expected_burn


@dataclass
class FitnessConfig:
    forest_grid: np.ndarray
    ignition_strategy: Strategy = "random"
    ignition_samples: int = 8
    ignition_seed: int = 0
    ignition_point: tuple[int, int] | None = None  # Kept for compatibility, not used in current strategies

    # Scalar-fitness weights (GA only).
    w_survived: float = 1.0
    w_burned: float = 2.0
    w_cut: float = 0.5

    # "Fit enough" acceptance thresholds.
    min_survival_rate: float = 0.85   # of remaining trees, this fraction must survive
    max_burn_rate: float = 0.05       # of original trees, at most this fraction burns
    max_cut_rate: float = 0.30        # of original trees, at most this fraction is cut


@dataclass
class FitnessReport:
    trees_original: int
    trees_cut: int
    trees_remaining: int
    trees_burned: int
    trees_survived: int
    survival_rate: float
    burn_rate: float
    cut_rate: float
    scalar_fitness: float
    objectives: tuple[float, float]  # (survived_count, -burned_count) — both "maximize"

    def as_dict(self) -> dict:
        return {
            "trees_original": self.trees_original,
            "trees_cut": self.trees_cut,
            "trees_remaining": self.trees_remaining,
            "trees_burned": self.trees_burned,
            "trees_survived": self.trees_survived,
            "survival_rate": self.survival_rate,
            "burn_rate": self.burn_rate,
            "cut_rate": self.cut_rate,
            "scalar_fitness": self.scalar_fitness,
            "objectives": list(self.objectives),
        }




def evaluate(cut_mask: np.ndarray, config: FitnessConfig) -> FitnessReport:
    """Apply the cut, run a fire (per the configured ignition strategy), score."""
    grid = config.forest_grid
    if cut_mask.shape != grid.shape:
        raise ValueError("cut_mask shape mismatch")

    trees_original = int(grid.sum())
    # cut only counts if it removes a tree (cutting an empty cell is wasted).
    effective_cut = ((grid == 1) & (cut_mask.astype(np.int8) == 1)).astype(np.int8)
    trees_cut = int(effective_cut.sum())
    remaining_grid = (grid & (1 - cut_mask.astype(np.int8))).astype(np.int8)
    trees_remaining = int(remaining_grid.sum())

    sim = expected_burn(
        remaining_grid,
        strategy=config.ignition_strategy,
        samples=config.ignition_samples,
        seed=config.ignition_seed,
    )

    trees_burned = sim.burned
    trees_survived = trees_remaining - trees_burned

    survival_rate = trees_survived / trees_remaining if trees_remaining else 0.0
    burn_rate = trees_burned / trees_original if trees_original else 0.0
    cut_rate = trees_cut / trees_original if trees_original else 0.0

    scalar_fitness = (
        config.w_survived * trees_survived
        - config.w_burned * trees_burned
        - config.w_cut * trees_cut
    )

    return FitnessReport(
        trees_original=trees_original,
        trees_cut=trees_cut,
        trees_remaining=trees_remaining,
        trees_burned=trees_burned,
        trees_survived=trees_survived,
        survival_rate=survival_rate,
        burn_rate=burn_rate,
        cut_rate=cut_rate,
        scalar_fitness=scalar_fitness,
        objectives=(float(trees_survived), float(-trees_burned)),
    )


def is_fit_enough(report: FitnessReport, config: FitnessConfig) -> bool:
    """Did a single individual clear all three acceptance thresholds?"""
    return (
        report.survival_rate >= config.min_survival_rate
        and report.burn_rate <= config.max_burn_rate
        and report.cut_rate <= config.max_cut_rate
    )


def is_population_fit_enough(
    reports: Sequence[FitnessReport],
    config: FitnessConfig,
) -> int | None:
    """Index of the first individual in the population that is fit enough, or None."""
    for i, r in enumerate(reports):
        if is_fit_enough(r, config):
            return i
    return None


@dataclass
class ConvergenceTracker:
    """Detects stalled GA runs — best scalar fitness unchanged for `patience` gens."""

    patience: int = 20
    _best: float = field(default=float("-inf"), init=False)
    _stale: int = field(default=0, init=False)

    def update(self, best_fitness: float) -> bool:
        """Returns True if the run has converged (no improvement for `patience` gens)."""
        if best_fitness > self._best + 1e-9:
            self._best = best_fitness
            self._stale = 0
        else:
            self._stale += 1
        return self._stale >= self.patience
