"""NSGA-II: multi-objective optimizer for the (survived, -burned) trade-off.

Classic Deb et al. 2002. Steps per generation:

  1. Combine parent + offspring (2N individuals)
  2. Non-dominated sort into Pareto fronts F1, F2, ...
  3. Fill the next population from F1, F2, ... until full. The last front to be
     included is trimmed by crowding distance (preserve spread).
  4. Binary tournament with (rank, crowding) comparison for parent selection.

Both objectives are framed as "maximize": objectives = (survived, -burned).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .chromosome import (
    bit_flip_mutation,
    random_individual,
    uniform_crossover,
    bit_flip_mutation_patch,
    random_individual_patch,
    uniform_crossover_patch,
    expand_patch_chromosome,
)
from .fitness import FitnessConfig, FitnessReport, evaluate


@dataclass
class NSGA2Stats:
    generation: int
    front_size: int
    front_objectives: list[tuple[float, float]]


@dataclass
class NSGA2Result:
    pareto_front: list[np.ndarray]               # cut masks on the final front
    pareto_reports: list[FitnessReport]          # one report per front member
    population_reports: list[FitnessReport]      # ALL final-population reports
    history: list[NSGA2Stats]
    generations_run: int


def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """a dominates b iff a >= b in all and a > b in at least one (maximize)."""
    return (a[0] >= b[0] and a[1] >= b[1]) and (a[0] > b[0] or a[1] > b[1])


def _fast_non_dominated_sort(objectives: list[tuple[float, float]]) -> list[list[int]]:
    """Return list of fronts, each a list of indices into `objectives`."""
    n = len(objectives)
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    domination_count = [0] * n
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objectives[p], objectives[q]):
                dominated_by[p].append(q)
            elif _dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return [f for f in fronts if f]


def _crowding_distance(front: list[int], objectives: list[tuple[float, float]]) -> dict[int, float]:
    """Standard NSGA-II crowding distance for one front."""
    distance = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        for idx in front:
            distance[idx] = float("inf")
        return distance

    for m in (0, 1):
        sorted_front = sorted(front, key=lambda i: objectives[i][m])
        distance[sorted_front[0]] = float("inf")
        distance[sorted_front[-1]] = float("inf")
        m_min = objectives[sorted_front[0]][m]
        m_max = objectives[sorted_front[-1]][m]
        span = m_max - m_min
        if span == 0:
            continue
        for k in range(1, len(sorted_front) - 1):
            prev_val = objectives[sorted_front[k - 1]][m]
            next_val = objectives[sorted_front[k + 1]][m]
            distance[sorted_front[k]] += (next_val - prev_val) / span
    return distance


def _crowded_compare(
    a: int,
    b: int,
    rank: dict[int, int],
    crowding: dict[int, float],
) -> int:
    """Return the winner index (a or b) under crowded-comparison."""
    if rank[a] < rank[b]:
        return a
    if rank[b] < rank[a]:
        return b
    if crowding[a] > crowding[b]:
        return a
    return b


def run_nsga2(
    config: FitnessConfig,
    population_size: int = 80,
    max_generations: int = 80,
    selection_strategy: str = "tournament",
    tournament_size: int = 2,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.01,
    initial_cut_probability: float = 0.15,
    seed: int = 0,
    progress_callback: Callable[[NSGA2Stats], None] | None = None,
) -> NSGA2Result:
    rng = np.random.default_rng(seed)
    shape = config.forest_grid.shape

    # Spread initial cut probability across the population so the very first
    # front already covers a range of trade-offs. `initial_cut_probability`
    # acts as the midpoint of that spread.
    lo = max(0.01, initial_cut_probability * 0.2)
    hi = min(0.7, initial_cut_probability * 3.0)
    init_probs = np.linspace(lo, hi, population_size)
    population: list[np.ndarray] = [
        random_individual(shape, float(p), rng, forest_grid=config.forest_grid) for p in init_probs
    ]
    reports: list[FitnessReport] = [evaluate(ind, config) for ind in population]
    objectives: list[tuple[float, float]] = [r.objectives for r in reports]

    history: list[NSGA2Stats] = []

    for gen in range(max_generations):
        # Sort current population for selection.
        fronts = _fast_non_dominated_sort(objectives)
        rank: dict[int, int] = {}
        crowding: dict[int, float] = {}
        for r, front in enumerate(fronts):
            cd = _crowding_distance(front, objectives)
            for idx in front:
                rank[idx] = r
                crowding[idx] = cd[idx]

        # Record stats from the current Pareto front.
        f1 = fronts[0]
        stats = NSGA2Stats(
            generation=gen,
            front_size=len(f1),
            front_objectives=[objectives[i] for i in f1],
        )
        history.append(stats)
        if progress_callback is not None:
            progress_callback(stats)

        # Generate offspring via binary tournament + crossover + mutation.
        offspring: list[np.ndarray] = []
        while len(offspring) < population_size:
            i, j = rng.integers(0, population_size, size=2)
            p1 = _crowded_compare(int(i), int(j), rank, crowding)
            i, j = rng.integers(0, population_size, size=2)
            p2 = _crowded_compare(int(i), int(j), rank, crowding)
            if rng.random() < crossover_rate:
                child = uniform_crossover(population[p1], population[p2], rng)
            else:
                child = population[p1].copy()
            # Mutation
            child = bit_flip_mutation(child, mutation_rate, rng)
            offspring.append(child)

        offspring_reports = [evaluate(c, config) for c in offspring]
        offspring_objectives = [r.objectives for r in offspring_reports]

        # Combine parents + offspring (2N) and pick the best N.
        combined_pop = population + offspring
        combined_reports = reports + offspring_reports
        combined_objectives = objectives + offspring_objectives

        combined_fronts = _fast_non_dominated_sort(combined_objectives)
        new_pop: list[np.ndarray] = []
        new_reports: list[FitnessReport] = []
        new_objectives: list[tuple[float, float]] = []
        for front in combined_fronts:
            if len(new_pop) + len(front) <= population_size:
                for idx in front:
                    new_pop.append(combined_pop[idx])
                    new_reports.append(combined_reports[idx])
                    new_objectives.append(combined_objectives[idx])
            else:
                # Trim the last front by crowding distance (keep the most spread out).
                cd = _crowding_distance(front, combined_objectives)
                remaining = population_size - len(new_pop)
                sorted_by_cd = sorted(front, key=lambda i: cd[i], reverse=True)
                for idx in sorted_by_cd[:remaining]:
                    new_pop.append(combined_pop[idx])
                    new_reports.append(combined_reports[idx])
                    new_objectives.append(combined_objectives[idx])
                break

        population = new_pop
        reports = new_reports
        objectives = new_objectives

    # Final Pareto front extraction.
    final_fronts = _fast_non_dominated_sort(objectives)
    front_idx = final_fronts[0]
    pareto_front = [population[i] for i in front_idx]
    pareto_reports = [reports[i] for i in front_idx]

    # Sort the front by survived ascending (one axis), so the frontend can plot.
    order = sorted(range(len(pareto_reports)), key=lambda k: pareto_reports[k].objectives[0])
    pareto_front = [pareto_front[k] for k in order]
    pareto_reports = [pareto_reports[k] for k in order]

    return NSGA2Result(
        pareto_front=pareto_front,
        pareto_reports=pareto_reports,
        population_reports=reports,
        history=history,
        generations_run=len(history),
    )


def run_nsga2_patch(
    config: FitnessConfig,
    population_size: int = 80,
    max_generations: int = 80,
    selection_strategy: str = "tournament",
    tournament_size: int = 2,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.01,
    initial_cut_probability: float = 0.15,
    patch_size: int = 2,
    seed: int = 0,
    progress_callback: Callable[[NSGA2Stats], None] | None = None,
) -> NSGA2Result:
    """Run NSGA-II with patch-based chromosome representation.
    
    Each patch_size × patch_size square is treated as a single unit.
    If map size not divisible by patch_size, pad with cut trees.
    """
    rng = np.random.default_rng(seed)
    shape = config.forest_grid.shape

    # Spread initial cut probability across the population
    lo = max(0.01, initial_cut_probability * 0.2)
    hi = min(0.7, initial_cut_probability * 3.0)
    init_probs = np.linspace(lo, hi, population_size)
    population: list[np.ndarray] = [
        random_individual_patch(shape, float(p), rng, patch_size=patch_size) for p in init_probs
    ]
    
    # Expand to full grid for evaluation
    expanded_pop = [expand_patch_chromosome(ind, shape, patch_size=patch_size) for ind in population]
    reports: list[FitnessReport] = [evaluate(ind, config) for ind in expanded_pop]
    objectives: list[tuple[float, float]] = [r.objectives for r in reports]

    history: list[NSGA2Stats] = []

    for gen in range(max_generations):
        # Sort current population for selection.
        fronts = _fast_non_dominated_sort(objectives)
        rank: dict[int, int] = {}
        crowding: dict[int, float] = {}
        for r, front in enumerate(fronts):
            cd = _crowding_distance(front, objectives)
            for idx in front:
                rank[idx] = r
                crowding[idx] = cd[idx]

        # Record stats from the current Pareto front.
        f1 = fronts[0]
        stats = NSGA2Stats(
            generation=gen,
            front_size=len(f1),
            front_objectives=[objectives[i] for i in f1],
        )
        history.append(stats)
        if progress_callback is not None:
            progress_callback(stats)

        # Generate offspring via binary tournament + crossover + mutation (on patches).
        offspring: list[np.ndarray] = []
        while len(offspring) < population_size:
            i, j = rng.integers(0, population_size, size=2)
            p1 = _crowded_compare(int(i), int(j), rank, crowding)
            i, j = rng.integers(0, population_size, size=2)
            p2 = _crowded_compare(int(i), int(j), rank, crowding)
            if rng.random() < crossover_rate:
                child = uniform_crossover_patch(population[p1], population[p2], rng)
            else:
                child = population[p1].copy()
            # Mutation (on patch level)
            child = bit_flip_mutation_patch(child, mutation_rate, rng)
            offspring.append(child)

        # Expand offspring and evaluate
        expanded_offspring = [expand_patch_chromosome(ind, shape, patch_size=patch_size) for ind in offspring]
        offspring_reports = [evaluate(c, config) for c in expanded_offspring]
        offspring_objectives = [r.objectives for r in offspring_reports]

        # Combine parents + offspring (2N) and pick the best N.
        combined_pop = population + offspring
        combined_reports = reports + offspring_reports
        combined_objectives = objectives + offspring_objectives

        combined_fronts = _fast_non_dominated_sort(combined_objectives)
        new_pop: list[np.ndarray] = []
        new_reports: list[FitnessReport] = []
        new_objectives: list[tuple[float, float]] = []
        for front in combined_fronts:
            if len(new_pop) + len(front) <= population_size:
                for idx in front:
                    new_pop.append(combined_pop[idx])
                    new_reports.append(combined_reports[idx])
                    new_objectives.append(combined_objectives[idx])
            else:
                # Trim the last front by crowding distance (keep the most spread out).
                cd = _crowding_distance(front, combined_objectives)
                remaining = population_size - len(new_pop)
                sorted_by_cd = sorted(front, key=lambda i: cd[i], reverse=True)
                for idx in sorted_by_cd[:remaining]:
                    new_pop.append(combined_pop[idx])
                    new_reports.append(combined_reports[idx])
                    new_objectives.append(combined_objectives[idx])
                break

        population = new_pop
        reports = new_reports
        objectives = new_objectives

    # Final Pareto front extraction and expansion.
    final_fronts = _fast_non_dominated_sort(objectives)
    front_idx = final_fronts[0]
    
    pareto_front = []
    pareto_reports = []
    for idx in front_idx:
        expanded = expand_patch_chromosome(population[idx], shape, patch_size=patch_size)
        pareto_front.append(expanded)
        pareto_reports.append(reports[idx])

    # Sort the front by survived ascending (one axis), so the frontend can plot.
    order = sorted(range(len(pareto_reports)), key=lambda k: pareto_reports[k].objectives[0])
    pareto_front = [pareto_front[k] for k in order]
    pareto_reports = [pareto_reports[k] for k in order]

    return NSGA2Result(
        pareto_front=pareto_front,
        pareto_reports=pareto_reports,
        population_reports=reports,
        history=history,
        generations_run=len(history),
    )
