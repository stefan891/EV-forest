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

import numpy as np

from ev_forest.fitness import FitnessConfig, is_fit_enough
from ev_forest.forest import make_forest
from ev_forest.ga import run_ga
from ev_forest.ignition import expected_burn
from ev_forest.nsga2 import run_nsga2

RESULTS_FILE = Path(__file__).parent / "param_sweep.json"

PROBLEM = {"dim": 60, "pct": 70}
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


def _worst_case_burned(cut_mask: np.ndarray, config: FitnessConfig) -> int:
    """Apply cut_mask to forest and run worst-case fire, return burned tree count."""
    grid = config.forest_grid
    effective_cut = ((grid == 1) & (cut_mask.astype(np.int8) == 1)).astype(np.int8)
    remaining_grid = (grid & (1 - effective_cut)).astype(np.int8)
    
    # Run worst-case fire on the remaining forest
    result = expected_burn(
        remaining_grid,
        strategy="worst_case",
        samples=1,  # Not used for worst_case strategy
        seed=SEED,
    )
    return int(result.burned)


def _ga_run(config: FitnessConfig, pop: int, gen: int) -> dict:
    t0 = time.time()
    mutation_rate = 0.01
    crossover_rate = 0.9
    tournament_size = 3
    selection_strategy = "tournament"
    elitism = 2
    patience = 25
    initial_cut_probability = 0.1
    
    result = run_ga(
        config,
        population_size=pop,
        max_generations=gen,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_size=tournament_size,
        selection_strategy=selection_strategy,
        elitism=elitism,
        patience=patience,
        initial_cut_probability=initial_cut_probability,
        seed=SEED,
    )
    worst_burned = _worst_case_burned(result.best_individual, config)
    return {
        "seed": SEED,
        "population_size": pop,
        "max_generations": gen,
        "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate,
        "tournament_size": tournament_size,
        "selection_strategy": selection_strategy,
        "initial_cut_probability": initial_cut_probability,
        "stopped_reason": result.stopped_reason,
        "generations_run": result.generations_run,
        "runtime_seconds": round(time.time() - t0, 2),
        "is_fit_enough": is_fit_enough(result.best_report, config),
        "best_report": result.best_report.as_dict(),
        "worst_case_burned": worst_burned,
    }


def _nsga2_run(config: FitnessConfig, pop: int, gen: int) -> dict:
    t0 = time.time()
    mutation_rate = 0.01
    crossover_rate = 0.9
    tournament_size = 2
    selection_strategy = "tournament"
    initial_cut_probability = 0.15
    
    result = run_nsga2(
        config,
        population_size=pop,
        max_generations=gen,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_size=tournament_size,
        selection_strategy=selection_strategy,
        initial_cut_probability=initial_cut_probability,
        seed=SEED,
    )
    pareto_summary = []
    for individual, report in zip(result.pareto_front, result.pareto_reports):
        worst_burned = _worst_case_burned(individual, config)
        pareto_summary.append({
            "trees_survived": report.trees_survived,
            "trees_burned": report.trees_burned,
            "trees_cut": report.trees_cut,
            "worst_case_burned": worst_burned,
        })
    
    return {
        "seed": SEED,
        "population_size": pop,
        "max_generations": gen,
        "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate,
        "tournament_size": tournament_size,
        "selection_strategy": selection_strategy,
        "initial_cut_probability": initial_cut_probability,
        "generations_run": result.generations_run,
        "runtime_seconds": round(time.time() - t0, 2),
        "pareto_front_size": len(result.pareto_front),
        "pareto_summary": pareto_summary,
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
            
            # Build config dict with algorithm-specific parameters
            cfg = {
                "algorithm": alg,
                "population_size": pop,
                "max_generations": gen,
                "mutation_rate": run["mutation_rate"],
                "crossover_rate": run["crossover_rate"],
                "selection_strategy": run["selection_strategy"],
                "initial_cut_probability": run["initial_cut_probability"],
                "tournament_size": run["tournament_size"]
            }

            
            results[key] = {
                "config": cfg,
                "run": run,
            }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results) - 1} entries to {RESULTS_FILE}")
    print(f"Total runtime: {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
