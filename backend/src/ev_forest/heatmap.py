"""Procedural ignition-probability heatmap.

A heatmap is a 2D float array of per-cell ignition weights. Values are
non-negative and unitless — only relative magnitudes matter (the ignition
sampler normalizes weights to probabilities at use time).

The default generator places `n_hotspots` Gaussian bumps at seeded random
positions on top of a uniform baseline. With `n_hotspots == 0` the result
is flat and ignition sampling falls back to uniform — i.e. the original
pre-heatmap behavior, no special case needed elsewhere.

This module is grid-agnostic: it produces a weight for every cell, including
empties. Callers mask by tree-occupancy at sampling time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASELINE_WEIGHT: float = 1.0


@dataclass
class HeatmapConfig:
    rows: int
    cols: int
    n_hotspots: int = 3
    hotspot_strength: float = 6.0      # Gaussian peak height above baseline
    hotspot_sigma_frac: float = 0.10   # sigma as fraction of min(rows, cols)
    baseline: float = BASELINE_WEIGHT
    seed: int = 0


def make_heatmap(config: HeatmapConfig) -> np.ndarray:
    """Generate a `rows x cols` float32 weight grid with Gaussian hotspots."""
    rows, cols = config.rows, config.cols
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    weights = np.full((rows, cols), config.baseline, dtype=np.float32)
    if config.n_hotspots <= 0 or config.hotspot_strength <= 0:
        return weights

    sigma = max(1.0, min(rows, cols) * config.hotspot_sigma_frac)
    rng = np.random.default_rng(config.seed)
    centers_r = rng.integers(0, rows, size=config.n_hotspots)
    centers_c = rng.integers(0, cols, size=config.n_hotspots)

    rr, cc = np.indices((rows, cols), dtype=np.float32)
    two_sigma_sq = 2.0 * sigma * sigma
    for r0, c0 in zip(centers_r, centers_c):
        bump = config.hotspot_strength * np.exp(
            -((rr - float(r0)) ** 2 + (cc - float(c0)) ** 2) / two_sigma_sq
        )
        weights += bump.astype(np.float32)
    return weights


def serialize_heatmap(weights: np.ndarray) -> list[list[float]]:
    """JSON-friendly view (rounded — the wire doesn't need full float precision)."""
    return np.round(weights.astype(np.float64), 4).tolist()
