"""Single-objective genetic algorithm.

Uses the scalar fitness from `fitness.evaluate`. Stops on any of:
  1. An individual passes `is_fit_enough` (target met)
  2. `max_generations` reached
  3. `ConvergenceTracker` reports no improvement for `patience` generations

Returns a full run history so the frontend can plot the learning curve.

Supports configurable selection strategies: "tournament", "rank_based", "comma", "plus"
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
from .fitness import (
    ConvergenceTracker,
    FitnessConfig,
    FitnessReport,
    evaluate,
    is_fit_enough,
)


@dataclass
class GAStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_report: FitnessReport


@dataclass
class GAResult:
    best_individual: np.ndarray
    best_report: FitnessReport
    history: list[GAStats]
    generations_run: int
    stopped_reason: str  # "fit_enough" | "max_generations" | "converged"


def _tournament_pick(
    fitnesses: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> int:
    """Pick tournament_size random individuals, return index of the fittest."""
    contenders = rng.integers(0, len(fitnesses), size=tournament_size)
    winner = contenders[np.argmax(fitnesses[contenders])]
    return int(winner)


def _rank_based_pick(
    fitnesses: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Rank-based selection: exponential distribution over sorted fitnesses."""
    ranks = np.argsort(np.argsort(-fitnesses))  # Higher fitness = lower rank
    weights = np.exp(-ranks / len(fitnesses))
    weights /= weights.sum()
    return int(rng.choice(len(fitnesses), p=weights))


def run_ga(
    config: FitnessConfig,
    population_size: int = 80,
    max_generations: int = 100,
    selection_strategy: str = "tournament",
    tournament_size: int = 3,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.01,
    initial_cut_probability: float = 0.1,
    elitism: int = 2,
    patience: int = 25,
    seed: int = 0,
    progress_callback: Callable[[GAStats], None] | None = None,
) -> GAResult:
    """Run single-objective GA.
    
    selection_strategy options:
      - "tournament": Binary tournament (default, tournament_size applies)
      - "rank_based": Exponential rank-based selection
      - "comma": Comma strategy (μ,λ) — only offspring survive to next gen
      - "plus": Plus strategy (μ+λ) — parents + offspring compete
    """
    rng = np.random.default_rng(seed)
    shape = config.forest_grid.shape

    population: list[np.ndarray] = [
        random_individual(shape, initial_cut_probability, rng, forest_grid=config.forest_grid)
        for _ in range(population_size)
    ]
    reports: list[FitnessReport] = [evaluate(ind, config) for ind in population]
    fitnesses = np.array([r.scalar_fitness for r in reports])

    history: list[GAStats] = []
    tracker = ConvergenceTracker(patience=patience)
    stopped_reason = "max_generations"

    for gen in range(max_generations):
        best_idx = int(np.argmax(fitnesses))
        stats = GAStats(
            generation=gen,
            best_fitness=float(fitnesses[best_idx]),
            mean_fitness=float(fitnesses.mean()),
            best_report=reports[best_idx],
        )
        history.append(stats)
        if progress_callback is not None:
            progress_callback(stats)

        if is_fit_enough(reports[best_idx], config):
            stopped_reason = "fit_enough"
            break
        if tracker.update(stats.best_fitness):
            stopped_reason = "converged"
            break

        # Elitism: carry the top `elitism` individuals.
        elite_idx = np.argsort(fitnesses)[-elitism:][::-1] if elitism > 0 else []
        next_pop: list[np.ndarray] = [population[i].copy() for i in elite_idx]

        #Offspring selection 
        while len(next_pop) < population_size:
            # Selection
            if selection_strategy == "tournament":
                a = _tournament_pick(fitnesses, tournament_size, rng)
                b = _tournament_pick(fitnesses, tournament_size, rng)
            elif selection_strategy == "rank_based":
                a = _rank_based_pick(fitnesses, rng)
                b = _rank_based_pick(fitnesses, rng)
            else:
                # comma / plus: use tournament for now
                a = _tournament_pick(fitnesses, tournament_size, rng)
                b = _tournament_pick(fitnesses, tournament_size, rng)
            
            # Crossover
            if rng.random() < crossover_rate:
                child = uniform_crossover(population[a], population[b], rng)
            else:
                child = population[a].copy()
            
            # Mutation
            child = bit_flip_mutation(child, mutation_rate, rng)
            next_pop.append(child)

        population = next_pop
        reports = [evaluate(ind, config) for ind in population]
        fitnesses = np.array([r.scalar_fitness for r in reports])

    best_idx = int(np.argmax(fitnesses))
    return GAResult(
        best_individual=population[best_idx],
        best_report=reports[best_idx],
        history=history,
        generations_run=len(history),
        stopped_reason=stopped_reason,
    )


def run_ga_patch(
    config: FitnessConfig,
    population_size: int = 80,
    max_generations: int = 100,
    selection_strategy: str = "tournament",
    tournament_size: int = 3,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.01,
    initial_cut_probability: float = 0.1,
    elitism: int = 2,
    patience: int = 25,
    patch_size: int = 2,
    seed: int = 0,
    progress_callback: Callable[[GAStats], None] | None = None,
) -> GAResult:
    """Run single-objective GA with patch-based chromosome representation.
    
    Each patch_size × patch_size square is treated as a single unit.
    If map size not divisible by patch_size, pad with cut trees.
    
    selection_strategy options:
      - "tournament": Binary tournament (default, tournament_size applies)
      - "rank_based": Exponential rank-based selection
      - "comma": Comma strategy (μ,λ) — only offspring survive to next gen
      - "plus": Plus strategy (μ+λ) — parents + offspring compete
    """
    rng = np.random.default_rng(seed)
    shape = config.forest_grid.shape

    # Generate patch-based population
    population: list[np.ndarray] = [
        random_individual_patch(shape, initial_cut_probability, rng, patch_size=patch_size)
        for _ in range(population_size)
    ]
    
    # Expand patches to full grid for evaluation
    expanded_pop = [expand_patch_chromosome(ind, shape, patch_size=patch_size) for ind in population]
    reports: list[FitnessReport] = [evaluate(ind, config) for ind in expanded_pop]
    fitnesses = np.array([r.scalar_fitness for r in reports])

    history: list[GAStats] = []
    tracker = ConvergenceTracker(patience=patience)
    stopped_reason = "max_generations"

    for gen in range(max_generations):
        best_idx = int(np.argmax(fitnesses))
        stats = GAStats(
            generation=gen,
            best_fitness=float(fitnesses[best_idx]),
            mean_fitness=float(fitnesses.mean()),
            best_report=reports[best_idx],
        )
        history.append(stats)
        if progress_callback is not None:
            progress_callback(stats)

        if is_fit_enough(reports[best_idx], config):
            stopped_reason = "fit_enough"
            break
        if tracker.update(stats.best_fitness):
            stopped_reason = "converged"
            break

        # Elitism: carry the top `elitism` individuals.
        elite_idx = np.argsort(fitnesses)[-elitism:][::-1] if elitism > 0 else []
        next_pop: list[np.ndarray] = [population[i].copy() for i in elite_idx]

        while len(next_pop) < population_size:
            # Selection
            if selection_strategy == "tournament":
                a = _tournament_pick(fitnesses, tournament_size, rng)
                b = _tournament_pick(fitnesses, tournament_size, rng)
            elif selection_strategy == "rank_based":
                a = _rank_based_pick(fitnesses, rng)
                b = _rank_based_pick(fitnesses, rng)
            else:
                # comma / plus: use tournament for now
                a = _tournament_pick(fitnesses, tournament_size, rng)
                b = _tournament_pick(fitnesses, tournament_size, rng)
            
            # Crossover
            if rng.random() < crossover_rate:
                child = uniform_crossover_patch(population[a], population[b], rng)
            else:
                child = population[a].copy()
            
            # Mutation (on patch level)
            child = bit_flip_mutation_patch(child, mutation_rate, rng)
            next_pop.append(child)

        population = next_pop
        expanded_pop = [expand_patch_chromosome(ind, shape, patch_size=patch_size) for ind in population]
        reports = [evaluate(ind, config) for ind in expanded_pop]
        fitnesses = np.array([r.scalar_fitness for r in reports])

    best_idx = int(np.argmax(fitnesses))
    best_expanded = expand_patch_chromosome(population[best_idx], shape, patch_size=patch_size)
    return GAResult(
        best_individual=best_expanded,
        best_report=reports[best_idx],
        history=history,
        generations_run=len(history),
        stopped_reason=stopped_reason,
    )

