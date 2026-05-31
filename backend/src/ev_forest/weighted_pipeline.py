"""Heat-zone-aware fitness pipeline, isolated from the base modules.

This module exists to keep the heat-zone feature out of `fitness.py`,
`ignition.py`, `ga.py`, and `nsga2.py` — collaborators iterating on the
algorithms shouldn't have to merge around weight handling.

`WeightedPipeline` bundles per-cell ignition weights with a base
`FitnessConfig` and provides weighted versions of `random_points`,
`expected_burn`, and `evaluate`. To drive the existing GA / NSGA-II runners
with the weighted evaluator (instead of duplicating ~100 lines of optimizer
code), `run_ga` / `run_nsga2` enter a context that temporarily rebinds the
`evaluate` symbol that those modules imported at load time. The patch is
scoped to the `with` block and always restored — including on exceptions.

Contract assumed of the base evaluator: `evaluate(cut_mask, config) ->
FitnessReport`. If that signature ever changes, mirror it on
`WeightedPipeline.evaluate`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from .fitness import FitnessConfig, FitnessReport, protect_ignition
from .forest import TREE
from .ignition import fixed_point
from .simulator import SimulationResult, simulate_fire


@dataclass
class WeightedPipeline:
    """A base `FitnessConfig` + per-cell ignition weights, with weighted methods."""

    base_config: FitnessConfig
    weights: np.ndarray  # same shape as base_config.forest_grid, non-negative

    def __post_init__(self) -> None:
        if self.weights.shape != self.base_config.forest_grid.shape:
            raise ValueError(
                f"weights shape {self.weights.shape} != grid shape "
                f"{self.base_config.forest_grid.shape}"
            )

    def random_points(
        self,
        grid: np.ndarray,
        n: int = 1,
        seed: int = 0,
    ) -> list[tuple[int, int]]:
        """Like `ignition.random_points`, but biased by `self.weights`.

        Tree cells are sampled with probability proportional to their weight.
        If every candidate weight is zero, falls back to uniform sampling so
        the optimizer doesn't crash on a degenerate heatmap.
        """
        rng = np.random.default_rng(seed)
        tree_positions = np.argwhere(grid == TREE)
        if len(tree_positions) == 0:
            return []
        n_pick = min(n, len(tree_positions))
        w = self.weights[tree_positions[:, 0], tree_positions[:, 1]].astype(np.float64)
        total = w.sum()
        p = (w / total) if total > 0 else None
        idx = rng.choice(len(tree_positions), size=n_pick, replace=False, p=p)
        return [(int(r), int(c)) for r, c in tree_positions[idx]]

    def expected_burn(self, grid: np.ndarray) -> SimulationResult:
        """Mirror of `ignition.expected_burn`, weighted for random/worst_case."""
        cfg = self.base_config
        rows, cols = grid.shape
        if cfg.ignition_strategy == "fixed":
            return simulate_fire(grid, fixed_point(rows, cols, cfg.ignition_point))

        points = self.random_points(grid, n=cfg.ignition_samples, seed=cfg.ignition_seed)
        if not points:
            return simulate_fire(grid, [])

        results = [simulate_fire(grid, [p]) for p in points]
        if cfg.ignition_strategy == "worst_case":
            return max(results, key=lambda r: r.burned)

        # "random": average burn across samples; final_grid is from the last sample.
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

    def evaluate(
        self,
        cut_mask: np.ndarray,
        _config: FitnessConfig | None = None,
    ) -> FitnessReport:
        """Mirror of `fitness.evaluate` using `self.expected_burn`.

        The `_config` parameter exists so this method can stand in for
        `fitness.evaluate` during the patched run — the optimizers call
        `evaluate(individual, config)`. We ignore that config and use
        `self.base_config`; during a patched run, they are the same object.
        """
        cfg = self.base_config
        grid = cfg.forest_grid
        if cut_mask.shape != grid.shape:
            raise ValueError("cut_mask shape mismatch")

        cut_mask = protect_ignition(cut_mask, cfg)

        trees_original = int(grid.sum())
        effective_cut = ((grid == 1) & (cut_mask.astype(np.int8) == 1)).astype(np.int8)
        trees_cut = int(effective_cut.sum())
        remaining_grid = (grid & (1 - cut_mask.astype(np.int8))).astype(np.int8)
        trees_remaining = int(remaining_grid.sum())

        sim = self.expected_burn(remaining_grid)

        trees_burned = sim.burned
        trees_survived = trees_remaining - trees_burned

        survival_rate = trees_survived / trees_remaining if trees_remaining else 0.0
        burn_rate = trees_burned / trees_original if trees_original else 0.0
        cut_rate = trees_cut / trees_original if trees_original else 0.0

        scalar_fitness = (
            cfg.w_survived * trees_survived
            - cfg.w_burned * trees_burned
            - cfg.w_cut * trees_cut
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

    @contextmanager
    def _patched_evaluate(self):
        """Rebind the `evaluate` symbol that ga.py / nsga2.py imported at load time.

        `from .fitness import evaluate` gives each module its own bound reference;
        patching `fitness.evaluate` alone wouldn't affect them. Always restored.
        """
        from . import ga as ga_mod
        from . import nsga2 as nsga2_mod
        ga_orig = ga_mod.evaluate
        nsga2_orig = nsga2_mod.evaluate
        ga_mod.evaluate = self.evaluate
        nsga2_mod.evaluate = self.evaluate
        try:
            yield
        finally:
            ga_mod.evaluate = ga_orig
            nsga2_mod.evaluate = nsga2_orig

    def run_ga(self, **kwargs):
        """Run the base GA with `self.evaluate` substituted in."""
        from .ga import run_ga
        with self._patched_evaluate():
            return run_ga(self.base_config, **kwargs)

    def run_nsga2(self, **kwargs):
        """Run the base NSGA-II with `self.evaluate` substituted in."""
        from .nsga2 import run_nsga2
        with self._patched_evaluate():
            return run_nsga2(self.base_config, **kwargs)
