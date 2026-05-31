"""Genetic operators on cut-mask chromosomes.

A chromosome is a 2D int8 array where 1 = cut, 0 = keep. Same shape as the
forest grid. Direct 1-to-1 binary mapping: each bit represents the cut decision
for the corresponding forest cell.
"""

from __future__ import annotations

import numpy as np


def random_individual(
    shape: tuple[int, int],
    cut_probability: float,
    rng: np.random.Generator,
    forest_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Initial random cut mask — each cell is cut with `cut_probability`.
    
    If forest_grid is provided, only generates cuts where trees exist (forest_grid == 1).
    """
    random_cuts = (rng.random(shape) < cut_probability).astype(np.int8)
    
    if forest_grid is not None:
        # Only allow cuts where trees exist
        return random_cuts & forest_grid
    else:
        return random_cuts


def uniform_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """50/50 per-cell inheritance from either parent."""
    pick_a = rng.random(parent_a.shape) < 0.5
    return np.where(pick_a, parent_a, parent_b).astype(np.int8)


def bit_flip_mutation(
    individual: np.ndarray,
    mutation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Flip bits with given mutation rate.
    
    Since population is initialized respecting forest structure, 
    simple bit flips are always valid.
    """
    mutant = individual.copy()
    flip_mask = rng.random(individual.shape) < mutation_rate
    return np.where(flip_mask, 1 - mutant, mutant).astype(np.int8)


def flatten_chromosome(chromosome: np.ndarray) -> np.ndarray:
    """Convert 2D chromosome to 1D for inspyred."""
    return chromosome.flatten().astype(np.int8)


def unflatten_chromosome(flat: np.ndarray | list, shape: tuple[int, int]) -> np.ndarray:
    """Convert 1D inspyred chromosome back to 2D."""
    return np.array(flat, dtype=np.int8).reshape(shape)


