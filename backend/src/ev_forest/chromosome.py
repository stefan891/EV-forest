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


# ============================================================================
# Patch-based chromosome: treat square patches as single units
# ============================================================================


def random_individual_patch(
    shape: tuple[int, int],
    cut_probability: float,
    rng: np.random.Generator,
    patch_size: int = 2,
    forest_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Generate random patch-based cut mask.
    
    Each patch_size × patch_size square is treated as a single unit.
    If map size not divisible by patch_size, pad with cut trees.
    
    Returns: patch chromosome (coarse grid, one value per patch).
    """
    h, w = shape
    
    # Calculate padded size
    padded_h = ((h + patch_size - 1) // patch_size) * patch_size
    padded_w = ((w + patch_size - 1) // patch_size) * patch_size
    
    # Patch grid dimensions
    patch_h = padded_h // patch_size
    patch_w = padded_w // patch_size
    
    # Generate random patches
    patches = (rng.random((patch_h, patch_w)) < cut_probability).astype(np.int8)
    
    return patches


def expand_patch_chromosome(
    patches: np.ndarray,
    original_shape: tuple[int, int],
    patch_size: int = 2,
) -> np.ndarray:
    """Expand patch chromosome back to full grid.
    
    Each patch value is replicated to fill its square region.
    Padding areas (if needed) are set to 1 (cut).
    """
    h, w = original_shape
    padded_h = ((h + patch_size - 1) // patch_size) * patch_size
    padded_w = ((w + patch_size - 1) // patch_size) * patch_size
    
    # Expand patches to padded size
    expanded = np.zeros((padded_h, padded_w), dtype=np.int8)
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            expanded[
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size,
            ] = patches[i, j]
    
    # Crop to original size and pad with cut trees
    result = expanded[:h, :w].copy()
    
    # Pad right edge if needed
    if w < padded_w:
        result = np.pad(result, ((0, 0), (0, padded_w - w)), constant_values=1)
    
    # Pad bottom edge if needed
    if h < padded_h:
        result = np.pad(result, ((0, padded_h - h), (0, 0)), constant_values=1)
    
    return result[:h, :w]  # Final crop to original size


def uniform_crossover_patch(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """50/50 per-patch inheritance from either parent."""
    pick_a = rng.random(parent_a.shape) < 0.5
    return np.where(pick_a, parent_a, parent_b).astype(np.int8)


def bit_flip_mutation_patch(
    patches: np.ndarray,
    mutation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Flip patch bits with given mutation rate.
    
    Each patch is flipped independently (affects entire patch at once).
    """
    mutant = patches.copy()
    flip_mask = rng.random(patches.shape) < mutation_rate
    return np.where(flip_mask, 1 - mutant, mutant).astype(np.int8)


