"""Flask backend for the Svelte frontend.

Endpoints (all JSON):

  POST /api/forest    -> generate a forest grid
  POST /api/simulate  -> run a fire on a posted grid, return per-step expansion
  POST /api/optimize/ga      -> run single-objective GA
  POST /api/optimize/nsga2   -> run NSGA-II, return Pareto front

Grids travel as nested lists of 0/1; on the wire the simulator wraps them in
numpy arrays. The Svelte client animates the per-step burn list locally.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from .fitness import FitnessConfig, evaluate
from .forest import make_forest, serialize
from .ga import run_ga
from .ignition import Strategy
from .nsga2 import run_nsga2
from .simulator import simulate_fire


def _as_grid(data: list[list[int]]) -> np.ndarray:
    arr = np.array(data, dtype=np.int8)
    if arr.ndim != 2:
        raise ValueError("grid must be 2D")
    return arr


def _fitness_config_from_payload(payload: dict[str, Any], grid: np.ndarray) -> FitnessConfig:
    ip = payload.get("ignition_point")
    ignition_point = (int(ip[0]), int(ip[1])) if ip else None
    return FitnessConfig(
        forest_grid=grid,
        ignition_strategy=payload.get("ignition_strategy", "fixed"),
        ignition_samples=int(payload.get("ignition_samples", 8)),
        ignition_seed=int(payload.get("ignition_seed", 0)),
        ignition_point=ignition_point,
        w_survived=float(payload.get("w_survived", 1.0)),
        w_burned=float(payload.get("w_burned", 2.0)),
        w_cut=float(payload.get("w_cut", 0.5)),
        min_survival_rate=float(payload.get("min_survival_rate", 0.85)),
        max_burn_rate=float(payload.get("max_burn_rate", 0.05)),
        max_cut_rate=float(payload.get("max_cut_rate", 0.30)),
    )


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.get("/api/health")
    def health() -> Any:
        return jsonify({"ok": True})

    @app.post("/api/forest")
    def forest_endpoint() -> Any:
        body = request.get_json(force=True) or {}
        forest = make_forest(
            rows=int(body.get("rows", 30)),
            cols=int(body.get("cols", 30)),
            layout=body.get("layout", "dense"),
            density=float(body.get("density", 1.0)),
            seed=int(body.get("seed", 0)),
        )
        return jsonify({
            "grid": serialize(forest.grid),
            "layout": forest.layout,
            "seed": forest.seed,
            "density": forest.density,
            "tree_count": forest.tree_count,
        })

    @app.post("/api/simulate")
    def simulate_endpoint() -> Any:
        body = request.get_json(force=True) or {}
        grid = _as_grid(body["grid"])
        cut = body.get("cut_mask")
        if cut is not None:
            cut_arr = _as_grid(cut)
            grid = (grid & (1 - cut_arr)).astype(np.int8)
        ignition = [(int(r), int(c)) for r, c in body.get("ignition", [])]
        if not ignition:
            ip = body.get("ignition_point")
            if ip is not None:
                ignition = [(int(ip[0]), int(ip[1]))]
            else:
                rows, cols = grid.shape
                ignition = [(rows // 2, cols // 2)]
        result = simulate_fire(grid, ignition)
        return jsonify({
            "total_trees": result.total_trees,
            "burned": result.burned,
            "survived": result.survived,
            "steps": result.steps,
            "burned_per_step": [
                [[int(r), int(c)] for r, c in step] for step in result.burned_per_step
            ],
            "final_grid": serialize(result.final_grid),
        })

    @app.post("/api/optimize/ga")
    def optimize_ga_endpoint() -> Any:
        body = request.get_json(force=True) or {}
        grid = _as_grid(body["grid"])
        config = _fitness_config_from_payload(body, grid)
        result = run_ga(
            config,
            population_size=int(body.get("population_size", 60)),
            max_generations=int(body.get("max_generations", 60)),
            selection_strategy=body.get("selection_strategy", "tournament"),
            tournament_size=int(body.get("tournament_size", 3)),
            crossover_rate=float(body.get("crossover_rate", 0.9)),
            mutation_rate=float(body.get("mutation_rate", 0.01)),
            initial_cut_probability=float(body.get("initial_cut_probability", 0.1)),
            elitism=int(body.get("elitism", 2)),
            patience=int(body.get("patience", 25)),
            seed=int(body.get("seed", 0)),
        )
        baseline = evaluate(np.zeros_like(grid), config)
        return jsonify({
            "best_cut_mask": serialize(result.best_individual),
            "best_report": result.best_report.as_dict(),
            "baseline": baseline.as_dict(),
            "stopped_reason": result.stopped_reason,
            "generations_run": result.generations_run,
            "history": [
                {
                    "generation": h.generation,
                    "best_fitness": h.best_fitness,
                    "mean_fitness": h.mean_fitness,
                    "best_report": h.best_report.as_dict(),
                }
                for h in result.history
            ],
        })

    @app.post("/api/optimize/nsga2")
    def optimize_nsga2_endpoint() -> Any:
        body = request.get_json(force=True) or {}
        grid = _as_grid(body["grid"])
        config = _fitness_config_from_payload(body, grid)
        result = run_nsga2(
            config,
            population_size=int(body.get("population_size", 60)),
            max_generations=int(body.get("max_generations", 50)),
            selection_strategy=body.get("selection_strategy", "tournament"),
            tournament_size=int(body.get("tournament_size", 2)),
            crossover_rate=float(body.get("crossover_rate", 0.9)),
            mutation_rate=float(body.get("mutation_rate", 0.01)),
            initial_cut_probability=float(body.get("initial_cut_probability", 0.15)),
            seed=int(body.get("seed", 0)),
        )
        baseline = evaluate(np.zeros_like(grid), config)
        return jsonify({
            "pareto_front": [
                {
                    "cut_mask": serialize(mask),
                    "report": report.as_dict(),
                }
                for mask, report in zip(result.pareto_front, result.pareto_reports)
            ],
            "population": [r.as_dict() for r in result.population_reports],
            "baseline": baseline.as_dict(),
            "generations_run": result.generations_run,
            "history": [
                {
                    "generation": h.generation,
                    "front_size": h.front_size,
                    "front_objectives": [list(o) for o in h.front_objectives],
                }
                for h in result.history
            ],
        })

    return app


def main() -> None:
    """Entry point for `uv run ev-forest` — starts the dev server on :5000."""
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
