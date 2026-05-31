"""Render heatmaps from `param_sweep.json` (pinned-problem parameter sweep).

Produces three PNGs in `tests/plots/`:
  - param_survivors.png  : best survivors as f(pop, gen) for GA and NSGA-II
  - param_runtime.png    : runtime (s)        as f(pop, gen) for GA and NSGA-II
  - param_compare.png    : NSGA-II survivors - GA survivors (diverging)

Run from `backend/` after `run_param_sweep.py`:

    uv run python tests/plot_param_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = Path(__file__).parent / "param_sweep.json"
PLOTS_DIR = Path(__file__).parent / "plots"


def _parse_key(key: str) -> tuple[str, int, int]:
    """'ga-pop60-gen50' -> ('ga', 60, 50)."""
    alg, pop_part, gen_part = key.split("-")
    return alg, int(pop_part.removeprefix("pop")), int(gen_part.removeprefix("gen"))


def _best_survivors(entry: dict) -> int:
    run = entry["run"]
    if "best_report" in run:
        return run["best_report"]["trees_survived"]
    if run.get("pareto_summary"):
        return max(p["trees_survived"] for p in run["pareto_summary"])
    return 0


def _runtime(entry: dict) -> float:
    return float(entry["run"]["runtime_seconds"])


def _matrix_for(entries: dict, alg: str, metric_fn,
                pops: list[int], gens: list[int]) -> np.ndarray:
    m = np.zeros((len(gens), len(pops)))
    for i, g in enumerate(gens):
        for j, p in enumerate(pops):
            m[i, j] = metric_fn(entries[f"{alg}-pop{p}-gen{g}"])
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


def _plot_pair(entries: dict, pops: list[int], gens: list[int], metric_fn,
               fmt: str, cmap: str, suptitle: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, alg, label in zip(axes, ("ga", "nsga2"), ("GA", "NSGA-II")):
        m = _matrix_for(entries, alg, metric_fn, pops, gens)
        im = ax.imshow(m, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(np.arange(len(pops)))
        ax.set_xticklabels(pops)
        ax.set_yticks(np.arange(len(gens)))
        ax.set_yticklabels(gens)
        ax.set_xlabel("population")
        ax.set_ylabel("generations")
        ax.set_title(label)
        _annotate(ax, m, fmt)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def _plot_compare(entries: dict, pops: list[int], gens: list[int],
                  out_path: Path) -> None:
    ga = _matrix_for(entries, "ga", _best_survivors, pops, gens)
    n = _matrix_for(entries, "nsga2", _best_survivors, pops, gens)
    diff = n - ga
    bound = max(abs(diff.min()), abs(diff.max()), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(diff, cmap="RdBu", aspect="auto", origin="lower",
                   vmin=-bound, vmax=bound)
    ax.set_xticks(np.arange(len(pops)))
    ax.set_xticklabels(pops)
    ax.set_yticks(np.arange(len(gens)))
    ax.set_yticklabels(gens)
    ax.set_xlabel("population")
    ax.set_ylabel("generations")
    ax.set_title("NSGA-II best survivors - GA best survivors\n(blue = NSGA-II wins, red = GA wins)")
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            ax.text(j, i, f"{int(diff[i, j]):+d}", ha="center", va="center",
                    color="black", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label="survivors delta")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")
    


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    raw = json.loads(RESULTS_FILE.read_text())
    problem = raw.pop("_problem", {})
    entries = raw

    pops, gens = set(), set()
    for key in entries:
        _, p, g = _parse_key(key)
        pops.add(p); gens.add(g)
    pops = sorted(pops); gens = sorted(gens)

    problem_label = f"{problem.get('rows','?')}x{problem.get('cols','?')}, density {problem.get('density','?')}"

    _plot_pair(
        entries, pops, gens, _best_survivors, "{:.0f}", "YlGn",
        f"Best survivors by (pop, gen) — problem: {problem_label}",
        PLOTS_DIR / "param_survivors.png",
    )
    _plot_pair(
        entries, pops, gens, _runtime, "{:.1f}s", "YlOrRd",
        f"Runtime (s) by (pop, gen) — problem: {problem_label}",
        PLOTS_DIR / "param_runtime.png",
    )
    _plot_compare(entries, pops, gens, PLOTS_DIR / "param_compare.png")


if __name__ == "__main__":
    main()
