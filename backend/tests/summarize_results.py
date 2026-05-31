"""Summarize the experiment sweep in `results.json`.

Reads the JSON written by `run_experiments.py` and prints comparison tables
plus a per-combination GA-vs-NSGA-II head-to-head on best survived count.

Run from `backend/`:

    uv run python tests/summarize_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "results.json"


def _bar(value: float, width: int = 20) -> str:
    """ASCII bar of `width` cells representing `value` in [0, 1]."""
    filled = max(0, min(width, int(round(value * width))))
    return "#" * filled + "-" * (width - filled)


def _parse_id(key: str) -> tuple[int, int, str]:
    """'30x30-50-ga' -> (30, 50, 'ga')."""
    dim_part, pct_part, alg = key.split("-")
    rows = int(dim_part.split("x")[0])
    return rows, int(pct_part), alg


def _ga_row(key: str, entry: dict) -> dict:
    run = entry["runs"][0]
    rep = run["best_report"]
    return {
        "id": key,
        "trees_original": rep["trees_original"],
        "trees_cut": rep["trees_cut"],
        "trees_burned": rep["trees_burned"],
        "trees_survived": rep["trees_survived"],
        "survival_rate": rep["survival_rate"],
        "scalar_fitness": rep["scalar_fitness"],
        "generations_run": run["generations_run"],
        "stopped_reason": run["stopped_reason"],
        "is_fit_enough": run["is_fit_enough"],
        "runtime_seconds": run["runtime_seconds"],
    }


def _nsga2_row(key: str, entry: dict) -> dict:
    run = entry["runs"][0]
    summary = run["pareto_summary"]
    if not summary:
        return {
            "id": key, "front_size": 0, "max_survived": 0,
            "min_burned": 0, "min_cut": 0,
            "runtime_seconds": run["runtime_seconds"],
        }
    return {
        "id": key,
        "front_size": run["pareto_front_size"],
        "max_survived": max(p["trees_survived"] for p in summary),
        "min_burned": min(p["trees_burned"] for p in summary),
        "min_cut": min(p["trees_cut"] for p in summary),
        "runtime_seconds": run["runtime_seconds"],
    }


def _print_ga_table(rows: list[dict]) -> None:
    print("=== GA results ===")
    print(f"{'ID':<18} {'trees':>6} {'cut':>5} {'burn':>5} {'surv':>6} "
          f"{'surv%':>7} {'fitness':>9} {'gens':>5} {'stopped':<18} {'time':>7}")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['id']:<18} {r['trees_original']:>6} {r['trees_cut']:>5} "
            f"{r['trees_burned']:>5} {r['trees_survived']:>6} "
            f"{r['survival_rate']*100:>6.1f}% {r['scalar_fitness']:>9.1f} "
            f"{r['generations_run']:>5} {r['stopped_reason']:<18} "
            f"{r['runtime_seconds']:>6.2f}s"
        )
    print()


def _print_nsga2_table(rows: list[dict]) -> None:
    print("=== NSGA-II Pareto fronts (best per criterion) ===")
    print(f"{'ID':<22} {'front':>5} {'best surv':>10} {'min burn':>9} "
          f"{'min cut':>8} {'time':>7}")
    print("-" * 75)
    for r in rows:
        print(
            f"{r['id']:<22} {r['front_size']:>5} {r['max_survived']:>10} "
            f"{r['min_burned']:>9} {r['min_cut']:>8} "
            f"{r['runtime_seconds']:>6.2f}s"
        )
    print()


def _print_head_to_head(ga_rows: list[dict], nsga2_rows: list[dict]) -> None:
    print("=== GA vs NSGA-II: best survived count ===")
    print(f"{'config':<14} {'GA surv':>8} {'NSGA-II':>8} {'winner':<9} "
          f"{'bar (NSGA-II share)':<22}")
    print("-" * 70)
    ga_by_combo = {r["id"].rsplit("-", 1)[0]: r for r in ga_rows}
    n_by_combo = {r["id"].rsplit("-", 1)[0]: r for r in nsga2_rows}
    for combo in sorted(set(ga_by_combo) & set(n_by_combo), key=_combo_sort):
        ga_val = ga_by_combo[combo]["trees_survived"]
        n_val = n_by_combo[combo]["max_survived"]
        winner = "GA" if ga_val > n_val else ("NSGA-II" if n_val > ga_val else "tie")
        share = n_val / (ga_val + n_val) if (ga_val + n_val) else 0.5
        print(f"{combo:<14} {ga_val:>8} {n_val:>8} {winner:<9} [{_bar(share)}]")
    print()


def _print_notes(ga_rows: list[dict], total_runtime: float) -> None:
    print("=== Notes ===")
    no_fit = [r["id"] for r in ga_rows if not r["is_fit_enough"]]
    if no_fit:
        print(f"- GA never reached fit_enough for: {', '.join(no_fit)}")
    else:
        print("- GA reached fit_enough on every combination.")
    slowest = max(ga_rows, key=lambda r: r["runtime_seconds"])
    print(f"- Slowest GA combo: {slowest['id']} at {slowest['runtime_seconds']:.2f}s")
    print(f"- Total experiment runtime captured: {total_runtime:.1f}s")


def _combo_sort(combo: str) -> tuple[int, int]:
    """Sort by dim, then percent — accepts '30x30-50' style."""
    dim_part, pct_part = combo.split("-")
    return int(dim_part.split("x")[0]), int(pct_part)


def _row_sort_key(r: dict) -> tuple[int, int]:
    rows, pct, _ = _parse_id(r["id"])
    return rows, pct


def main() -> None:
    data = json.loads(RESULTS_FILE.read_text())
    ga_rows: list[dict] = []
    nsga2_rows: list[dict] = []
    total_runtime = 0.0
    for key, entry in data.items():
        alg = key.split("-")[-1]
        for run in entry["runs"]:
            total_runtime += run.get("runtime_seconds", 0.0)
        if alg == "ga":
            ga_rows.append(_ga_row(key, entry))
        elif alg == "nsga2":
            nsga2_rows.append(_nsga2_row(key, entry))

    ga_rows.sort(key=_row_sort_key)
    nsga2_rows.sort(key=_row_sort_key)

    _print_ga_table(ga_rows)
    _print_nsga2_table(nsga2_rows)
    _print_head_to_head(ga_rows, nsga2_rows)
    _print_notes(ga_rows, total_runtime)


if __name__ == "__main__":
    main()
