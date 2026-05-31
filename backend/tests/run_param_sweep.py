"""Sweep (algorithm, population, generations) on a single pinned problem.

Pinned to a 60x60 random forest with density 0.7 — chosen because GA hits
`max_generations` on this problem at baseline pop=60/gen=50, so the budget
parameters actually matter.

Runs in parallel across `MAX_WORKERS` processes. Each run is independent
(same forest grid, same seed), so order of completion does not affect results.

Writes per-run results to `param_sweep.json` keyed by
`{alg}-pop{pop}-gen{gen}`. Run from `backend/`:

    uv run python tests/run_param_sweep.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ev_forest.fitness import FitnessConfig, is_fit_enough
from ev_forest.forest import make_forest
from ev_forest.ga import run_ga
from ev_forest.nsga2 import run_nsga2

RESULTS_FILE = Path(__file__).parent / "param_sweep.json"

PROBLEM = {"dim": 50, "pct": 70}
ALGORITHMS = ["ga", "nsga2"]
POPULATIONS = [20, 60, 120]
GENERATIONS = [20, 50, 100]
SEED = 0
MAX_WORKERS = 4


def _make_id(alg: str, pop: int, gen: int) -> str:
    return f"{alg}-pop{pop}-gen{gen}"


def _build_config() -> FitnessConfig:
    forest = make_forest(
        rows=PROBLEM["dim"], cols=PROBLEM["dim"],
        layout="random", density=PROBLEM["pct"] / 100.0, seed=SEED,
    )
    return FitnessConfig(
        forest_grid=forest.grid,
        ignition_strategy="random",
        ignition_samples=8,
        ignition_seed=SEED,
    )


def _ga_run(config: FitnessConfig, pop: int, gen: int) -> dict:
    t0 = time.time()
    result = run_ga(config, population_size=pop, max_generations=gen, seed=SEED)
    return {
        "seed": SEED,
        "population_size": pop,
        "max_generations": gen,
        "stopped_reason": result.stopped_reason,
        "generations_run": result.generations_run,
        "runtime_seconds": round(time.time() - t0, 2),
        "is_fit_enough": is_fit_enough(result.best_report, config),
        "best_report": result.best_report.as_dict(),
    }


def _nsga2_run(config: FitnessConfig, pop: int, gen: int) -> dict:
    t0 = time.time()
    result = run_nsga2(config, population_size=pop, max_generations=gen, seed=SEED)
    return {
        "seed": SEED,
        "population_size": pop,
        "max_generations": gen,
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


def _execute(args: tuple[str, int, int, FitnessConfig]) -> tuple[str, int, int, dict]:
    """Worker entry point. Runs one GA or NSGA-II configuration."""
    alg, pop, gen, config = args
    run = _ga_run(config, pop, gen) if alg == "ga" else _nsga2_run(config, pop, gen)
    return alg, pop, gen, run


def main() -> None:
    config = _build_config()
    results: dict[str, dict] = {
        "_problem": {
            "rows": PROBLEM["dim"],
            "cols": PROBLEM["dim"],
            "layout": "random",
            "density": PROBLEM["pct"] / 100.0,
            "ignition_strategy": "random",
            "ignition_samples": 8,
            "seed": SEED,
        },
    }
    # Schedule longest-first: pop*gen as a proxy for cost. Reduces makespan.
    tasks = sorted(
        ((alg, pop, gen, config) for alg in ALGORITHMS
         for pop in POPULATIONS for gen in GENERATIONS),
        key=lambda t: -(t[1] * t[2]),
    )
    total = len(tasks)
    done = 0
    overall_start = time.time()
    print(f"Dispatching {total} runs across {MAX_WORKERS} workers...", flush=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_execute, t): t for t in tasks}
        for fut in as_completed(futures):
            alg, pop, gen, run = fut.result()
            done += 1
            key = _make_id(alg, pop, gen)
            print(f"[{done}/{total}] {key} done in {run['runtime_seconds']}s", flush=True)
            results[key] = {
                "config": {
                    "algorithm": alg,
                    "population_size": pop,
                    "max_generations": gen,
                },
                "run": run,
            }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results) - 1} entries to {RESULTS_FILE}")
    print(f"Total runtime: {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
