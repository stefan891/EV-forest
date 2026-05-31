"""Render heatmaps from `operators_sweep.json`.

Produces two PNGs in `tests/plots/`:
  - operators_survivors.png : best survivors per (mutation, crossover); one
    panel per tournament_size for GA plus one panel for NSGA-II.
  - operators_runtime.png   : runtime in seconds, same layout.

Run from `backend/` after `run_operators_sweep.py`:

    uv run python tests/plot_operators_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = Path(__file__).parent / "operators_sweep.json"
PLOTS_DIR = Path(__file__).parent / "plots"


def _best_survivors(entry: dict) -> int:
    run = entry["run"]
    if "best_report" in run:
        return run["best_report"]["trees_survived"]
    if run.get("pareto_summary"):
        return max(p["trees_survived"] for p in run["pareto_summary"])
    return 0


def _runtime(entry: dict) -> float:
    return float(entry["run"]["runtime_seconds"])


def _matrix(entries: dict, alg: str, mut_vals: list[float], xover_vals: list[float],
            tour: int | None, metric_fn) -> np.ndarray:
    m = np.zeros((len(mut_vals), len(xover_vals)))
    for i, mut in enumerate(mut_vals):
        for j, xover in enumerate(xover_vals):
            if alg == "ga":
                key = f"ga-mut{mut}-xover{xover}-tour{tour}"
            else:
                key = f"nsga2-mut{mut}-xover{xover}"
            m[i, j] = metric_fn(entries[key])
    return m


def _annotate(ax, matrix: np.ndarray, fmt: str) -> None:
    vmin, vmax = matrix.min(), matrix.max()
    span = max(vmax - vmin, 1e-9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            normalized = (matrix[i, j] - vmin) / span
            color = "white" if normalized > 0.55 else "black"
            ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center",
                    color=color, fontsize=10)


def _draw_panel(ax, matrix: np.ndarray, mut_vals: list[float], xover_vals: list[float],
                title: str, cmap: str, fmt: str) -> None:
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(xover_vals)))
    ax.set_xticklabels(xover_vals)
    ax.set_yticks(np.arange(len(mut_vals)))
    ax.set_yticklabels(mut_vals)
    ax.set_xlabel("crossover rate")
    ax.set_ylabel("mutation rate")
    ax.set_title(title)
    _annotate(ax, matrix, fmt)
    plt.colorbar(im, ax=ax, shrink=0.85)


def _plot_all(entries: dict, mut_vals: list[float], xover_vals: list[float],
              tour_vals: list[int], metric_fn, fmt: str, cmap: str,
              suptitle: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, tour in zip(axes[:3], tour_vals):
        m = _matrix(entries, "ga", mut_vals, xover_vals, tour, metric_fn)
        _draw_panel(ax, m, mut_vals, xover_vals,
                    f"GA — tournament_size={tour}", cmap, fmt)
    m = _matrix(entries, "nsga2", mut_vals, xover_vals, None, metric_fn)
    _draw_panel(axes[3], m, mut_vals, xover_vals,
                "NSGA-II (fixed selection)", cmap, fmt)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    raw = json.loads(RESULTS_FILE.read_text())
    problem = raw.pop("_problem", {})
    entries = raw

    mut_vals = sorted({e["config"]["mutation_rate"] for e in entries.values()})
    xover_vals = sorted({e["config"]["crossover_rate"] for e in entries.values()})
    tour_vals = sorted({e["config"]["tournament_size"]
                        for e in entries.values()
                        if "tournament_size" in e["config"]})

    label = (f"{problem.get('rows','?')}x{problem.get('cols','?')}, "
             f"density {problem.get('density','?')}")

    _plot_all(entries, mut_vals, xover_vals, tour_vals, _best_survivors, "{:.0f}",
              "YlGn", f"Best survivors by operator params — problem: {label}",
              PLOTS_DIR / "operators_survivors.png")
    _plot_all(entries, mut_vals, xover_vals, tour_vals, _runtime, "{:.1f}s",
              "YlOrRd", f"Runtime (s) by operator params — problem: {label}",
              PLOTS_DIR / "operators_runtime.png")


if __name__ == "__main__":
    main()
