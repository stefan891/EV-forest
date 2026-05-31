"""Render charts from the experiment sweep in `results.json`.

Produces three PNGs in `tests/plots/`:
  - survivors.png   : GA vs NSGA-II best survived count per config
  - pareto.png      : NSGA-II Pareto front per config (subplots)
  - runtime.png     : GA vs NSGA-II wall-clock per config

Run from `backend/`:

    uv run python tests/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = Path(__file__).parent / "results.json"
PLOTS_DIR = Path(__file__).parent / "plots"


def _combo(key: str) -> str:
    """'30x30-50-ga' -> '30x30-50'."""
    return key.rsplit("-", 1)[0]


def _combo_sort(combo: str) -> tuple[int, int]:
    dim_part, pct_part = combo.split("-")
    return int(dim_part.split("x")[0]), int(pct_part)


def _split_runs(data: dict) -> tuple[dict, dict]:
    ga, nsga2 = {}, {}
    for key, entry in data.items():
        alg = key.split("-")[-1]
        if alg == "ga":
            ga[_combo(key)] = entry["runs"][0]
        elif alg == "nsga2":
            nsga2[_combo(key)] = entry["runs"][0]
    return ga, nsga2


def plot_survivors(ga: dict, nsga2: dict, out_path: Path) -> None:
    combos = sorted(set(ga) & set(nsga2), key=_combo_sort)
    ga_vals = [ga[c]["best_report"]["trees_survived"] for c in combos]
    nsga2_vals = [max(p["trees_survived"] for p in nsga2[c]["pareto_summary"]) for c in combos]

    x = np.arange(len(combos))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_ga = ax.bar(x - width / 2, ga_vals, width, label="GA", color="#2563eb")
    bars_n = ax.bar(x + width / 2, nsga2_vals, width, label="NSGA-II (best on front)", color="#dc2626")

    ax.set_xticks(x)
    ax.set_xticklabels(combos)
    ax.set_ylabel("Trees survived")
    ax.set_title("GA vs NSGA-II — best surviving trees per config")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for bars in (bars_ga, bars_n):
        ax.bar_label(bars, fontsize=8, padding=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_pareto(nsga2: dict, out_path: Path) -> None:
    combos = sorted(nsga2, key=_combo_sort)
    cols = 3
    rows = (len(combos) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)

    for ax, combo in zip(axes.flat, combos):
        summary = nsga2[combo]["pareto_summary"]
        cuts = [p["trees_cut"] for p in summary]
        survived = [p["trees_survived"] for p in summary]
        burned = [p["trees_burned"] for p in summary]
        order = np.argsort(cuts)
        cuts_o = np.array(cuts)[order]
        surv_o = np.array(survived)[order]
        scatter = ax.scatter(cuts_o, surv_o, c=np.array(burned)[order], cmap="Reds",
                             s=60, edgecolors="black", linewidths=0.5, zorder=3)
        ax.plot(cuts_o, surv_o, color="#94a3b8", linestyle="--", linewidth=1, zorder=2)
        ax.set_title(combo)
        ax.set_xlabel("trees cut")
        ax.set_ylabel("trees survived")
        ax.grid(linestyle=":", alpha=0.5)
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.85)
        cbar.set_label("burned", fontsize=8)

    for ax in axes.flat[len(combos):]:
        ax.axis("off")  

    fig.suptitle("NSGA-II Pareto fronts — survived vs cut (color = burned)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_runtime(ga: dict, nsga2: dict, out_path: Path) -> None:
    combos = sorted(set(ga) & set(nsga2), key=_combo_sort)
    ga_time = [ga[c]["runtime_seconds"] for c in combos]
    nsga2_time = [nsga2[c]["runtime_seconds"] for c in combos]

    x = np.arange(len(combos))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_ga = ax.bar(x - width / 2, ga_time, width, label="GA", color="#2563eb")
    bars_n = ax.bar(x + width / 2, nsga2_time, width, label="NSGA-II", color="#dc2626")

    ax.set_xticks(x)
    ax.set_xticklabels(combos)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Wall-clock runtime per config")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for bars in (bars_ga, bars_n):
        ax.bar_label(bars, fmt="%.2fs", fontsize=8, padding=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    data = json.loads(RESULTS_FILE.read_text())
    ga, nsga2 = _split_runs(data)
    plot_survivors(ga, nsga2, PLOTS_DIR / "survivors.png")
    plot_pareto(nsga2, PLOTS_DIR / "pareto.png")
    plot_runtime(ga, nsga2, PLOTS_DIR / "runtime.png")


if __name__ == "__main__":
    main()
