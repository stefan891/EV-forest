"""Sweep GA + NSGA-II across map dimensions and forest randomness levels.

Writes a summary per `(dim, randomness%, algorithm)` combination into
`results.json`, keyed by `{dim}x{dim}-{pct}-{alg}`. Run from `backend/`:

    uv run python tests/run_experiments.py

Ignites with the 'random' strategy (8 samples) so the optimizer faces a
robust-defense problem rather than a single fixed spark.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ev_forest.fitness import FitnessConfig, is_fit_enough
from ev_forest.forest import make_forest
from ev_forest.ga import run_ga
from ev_forest.nsga2 import run_nsga2

RESULTS_FILE = Path(__file__).parent / "results.json"

DIMENSIONS = [30, 60]
RANDOMNESS_PERCENTAGES = [30, 50, 70]
ALGORITHMS = ["ga", "nsga2"]
SEEDS = [0]


def _make_id(dim: int, pct: int, alg: str) -> str:
    return f"{dim}x{dim}-{pct}-{alg}"


def _config_for(rows: int, cols: int, density: float, seed: int) -> FitnessConfig:
    forest = make_forest(rows=rows, cols=cols, layout="random", density=density, seed=seed)
    return FitnessConfig(
        forest_grid=forest.grid,
        ignition_strategy="random",
        ignition_samples=8,
        ignition_seed=seed,
    )


def _ga_run(config: FitnessConfig, seed: int) -> dict:
    t0 = time.time()
    result = run_ga(config, population_size=60, max_generations=50, seed=seed)
    return {
        "seed": seed,
        "stopped_reason": result.stopped_reason,
        "generations_run": result.generations_run,
        "runtime_seconds": round(time.time() - t0, 2),
        "is_fit_enough": is_fit_enough(result.best_report, config),
        "best_report": result.best_report.as_dict(),
    }


def _nsga2_run(config: FitnessConfig, seed: int) -> dict:
    t0 = time.time()
    result = run_nsga2(config, population_size=60, max_generations=40, seed=seed)
    return {
        "seed": seed,
        "generations_run": result.generations_run,
        "runtime_seconds": round(time.time() - t0, 2),
        "pareto_front_size": len(result.pareto_front),
        "pareto_summary": [
            {
                "trees_survived": r.trees_survived,
                "trees_burned": r.trees_burned,
                "trees_cut": r.trees_cut,
            }
            for r in result.pareto_reports
        ],
    }


def main() -> None:
    results: dict[str, dict] = {}
    total = len(DIMENSIONS) * len(RANDOMNESS_PERCENTAGES) * len(ALGORITHMS) * len(SEEDS)
    done = 0
    overall_start = time.time()

    for dim in DIMENSIONS:
        for pct in RANDOMNESS_PERCENTAGES:
            density = pct / 100.0
            for alg in ALGORITHMS:
                key = _make_id(dim, pct, alg)
                runs = []
                for seed in SEEDS:
                    done += 1
                    print(f"[{done}/{total}] {key} (seed={seed}) ...", flush=True)
                    config = _config_for(dim, dim, density, seed)
                    runs.append(_ga_run(config, seed) if alg == "ga" else _nsga2_run(config, seed))
                results[key] = {
                    "config": {
                        "rows": dim,
                        "cols": dim,
                        "layout": "random",
                        "density": density,
                        "algorithm": alg,
                    },
                    "runs": runs,
                }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} entries to {RESULTS_FILE}")
    print(f"Total runtime: {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
