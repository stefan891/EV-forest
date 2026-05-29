"""Genetic operators on cut-mask chromosomes.

A chromosome is a 2D int8 array where 1 = cut, 0 = keep. Same shape as the
forest grid. These operators are shared between the single-objective GA and
NSGA-II.
"""

from __future__ import annotations

import numpy as np


def random_individual(
    shape: tuple[int, int],
    cut_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initial random cut mask — each cell is cut with `cut_probability`."""
    return (rng.random(shape) < cut_probability).astype(np.int8)


def uniform_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """50/50 per-cell inheritance from either parent."""
    pick_a = rng.random(parent_a.shape) < 0.5
    child = np.where(pick_a, parent_a, parent_b).astype(np.int8)
    return child


def bit_flip_mutation(
    individual: np.ndarray,
    mutation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Each cell flips with probability `mutation_rate`. Returns a new array."""
    flip_mask = rng.random(individual.shape) < mutation_rate
    return np.where(flip_mask, 1 - individual, individual).astype(np.int8)
