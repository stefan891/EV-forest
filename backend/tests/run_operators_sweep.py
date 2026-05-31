"""Sweep operator parameters on the pinned 60x60-70 problem.

GA: mutation_rate x crossover_rate x tournament_size = 27 runs
NSGA-II: mutation_rate x crossover_rate = 9 runs (parent selection is fixed
in NSGA-II — binary tournament with rank + crowding comparison).

Population and generation budgets are pinned to baseline values (GA: pop=60,
gen=50; NSGA-II: pop=60, gen=40) so the only varying factor is the operator
parameters.

Runs in parallel via `ProcessPoolExecutor` with `MAX_WORKERS=4`. Writes per-run
results to `operators_sweep.json` keyed by either
`ga-mut{m}-xover{c}-tour{t}` or `nsga2-mut{m}-xover{c}`. Run from `backend/`:

    uv run python tests/run_operators_sweep.py
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

RESULTS_FILE = Path(__file__).parent / "operators_sweep.json"

PROBLEM = {"dim": 20, "pct": 70} # 60, 70
GA_POP, GA_GEN = 50, 50 # 100, 50
NSGA2_POP, NSGA2_GEN = 100, 40 
SEED = 0
MAX_WORKERS = 2

MUTATION_RATES = [0.005, 0.01, 0.05]
CROSSOVER_RATES = [0.5, 0.7, 0.9]
TOURNAMENT_SIZES = [2, 5]


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


def _ga_run(config: FitnessConfig, mut: float, xover: float, tour: int | None = None, selection_strategy: str = "tournament") -> dict:
    t0 = time.time()
    kwargs = {
        "config": config,
        "population_size": GA_POP,
        "max_generations": GA_GEN,
        "mutation_rate": mut,
        "crossover_rate": xover,
        "selection_strategy": selection_strategy,
        "seed": SEED,
    }
    if tour is not None:
        kwargs["tournament_size"] = tour
    result = run_ga(**kwargs)
    worst_burned = _worst_case_burned(result.best_individual, config)
    run_data = {
        "seed": SEED,
        "mutation_rate": mut,
        "crossover_rate": xover,
        "population_size": GA_POP,
        "max_generations": GA_GEN,
        "stopped_reason": result.stopped_reason,
        "generations_run": result.generations_run,
        "runtime_seconds": round(time.time() - t0, 2),
        "is_fit_enough": is_fit_enough(result.best_report, config),
        "best_report": result.best_report.as_dict(),
        "worst_case_burned": worst_burned,
    }
    if tour is not None:
        run_data["tournament_size"] = tour
    return run_data


def _nsga2_run(config: FitnessConfig, mut: float, xover: float) -> dict:
    t0 = time.time()
    result = run_nsga2(
        config,
        population_size=NSGA2_POP, max_generations=NSGA2_GEN,
        mutation_rate=mut, crossover_rate=xover,
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
        "mutation_rate": mut,
        "crossover_rate": xover,
        "population_size": NSGA2_POP,
        "max_generations": NSGA2_GEN,
        "generations_run": result.generations_run,
        "runtime_seconds": round(time.time() - t0, 2),
        "pareto_front_size": len(result.pareto_front),
        "pareto_summary": pareto_summary,
    }


def _execute(args: tuple) -> tuple:
    """Worker entry point. `tour` is None for rank-based or NSGA-II."""
    alg, mut, xover, tour, selection_strategy, config = args
    if alg == "ga":
        run = _ga_run(config, mut, xover, tour, selection_strategy)
    else:
        run = _nsga2_run(config, mut, xover)
    return (alg, mut, xover, tour, selection_strategy), run


def _make_key(alg: str, mut: float, xover: float, tour: int | None, selection_strategy: str = "tournament") -> str:
    if alg == "ga":
        if selection_strategy == "rank_based":
            return f"ga-mut{mut}-xover{xover}-rank"
        return f"ga-mut{mut}-xover{xover}-tour{tour}"
    return f"nsga2-mut{mut}-xover{xover}"


def main() -> None:
    config = _build_config()
    tasks: list[tuple] = []
    for mut in MUTATION_RATES:
        for xover in CROSSOVER_RATES:
            # Tournament-based GA
            for tour in TOURNAMENT_SIZES:
                tasks.append(("ga", mut, xover, tour, "tournament", config))
            # Rank-based GA
            tasks.append(("ga", mut, xover, None, "rank_based", config))
            # NSGA-II
            tasks.append(("nsga2", mut, xover, None, "tournament", config))

    total = len(tasks)
    results: dict[str, dict] = {
        "_problem": {
            "rows": PROBLEM["dim"],
            "cols": PROBLEM["dim"],
            "layout": "random",
            "density": PROBLEM["pct"] / 100.0,
            "ignition_strategy": "random",
            "ignition_samples": 8,
            "seed": SEED,
            "ga_population": GA_POP,
            "ga_generations": GA_GEN,
            "nsga2_population": NSGA2_POP,
            "nsga2_generations": NSGA2_GEN,
        },
    }
    done = 0
    overall_start = time.time()
    print(f"Dispatching {total} runs across {MAX_WORKERS} workers...", flush=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_execute, t): t for t in tasks}
        for fut in as_completed(futures):
            (alg, mut, xover, tour, selection_strategy), run = fut.result()
            done += 1
            key = _make_key(alg, mut, xover, tour, selection_strategy)
            print(f"[{done}/{total}] {key} done in {run['runtime_seconds']}s", flush=True)
            cfg = {"algorithm": alg, "mutation_rate": mut, "crossover_rate": xover, "selection_strategy": selection_strategy}
            if tour is not None:
                cfg["tournament_size"] = tour
            results[key] = {"config": cfg, "run": run}

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results) - 1} entries to {RESULTS_FILE}")
    print(f"Total runtime: {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
