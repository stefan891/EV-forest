# EV-Forest

Bio-Inspired AI course project. Optimize forest cuts to minimize wildfire
spread while maximizing surviving trees, using a genetic algorithm and
NSGA-II for the multi-objective trade-off.

## Problem description

A forest is a 2D grid of trees. Fire spreads from any burning tree to its 8
neighbors (diagonals included) if they are also trees; a single empty cell
blocks the spread. The optimizer picks a *cut mask* — a binary array the same
size as the forest — that maximizes surviving trees and minimizes burned trees.
These two objectives conflict (cut everything → 0 burn but 0 survival), so
NSGA-II returns a Pareto front of trade-offs; a single-objective GA is provided
as a baseline.

## Layout

```
backend/                # Python: simulator + optimizers + Flask API
  src/ev_forest/
    forest.py           # grid generation (dense / baseline / random / checker)
    simulator.py        # deterministic 8-neighbor fire-spread BFS
    ignition.py         # ignition strategies: fixed / random / worst-case
    fitness.py          # evaluate + is_fit_enough + ConvergenceTracker
    chromosome.py       # random init, uniform crossover, bit-flip mutation
    ga.py               # single-objective genetic algorithm
    nsga2.py            # multi-objective NSGA-II
    server.py           # Flask app exposing /api/{forest,simulate,optimize/*}
  tests/                # pytest — pins the fire-spread invariants
frontend/               # Svelte + Vite: canvas grid + charts
  src/
    App.svelte          # main layout, state, API orchestration
    lib/
      ForestCanvas.svelte   # grid renderer, click-to-ignite, animated burn
      Controls.svelte       # forest gen + fitness thresholds + optimizer params
      ParetoChart.svelte    # NSGA-II Pareto front scatter
      HistoryChart.svelte   # GA best/mean fitness over generations
      api.js                # fetch wrapper for the backend
```

## Running

Prerequisites: `uv` (Python package manager) and Node.js 18+.

### Backend

```bash
cd backend
uv sync                          # install Python deps into .venv
uv run pytest tests/             # run the 17 simulator + optimizer tests
uv run ev-forest                 # start Flask dev server on http://127.0.0.1:5000
```

### Frontend

```bash
cd frontend
npm install                      # one-time
npm run dev                      # Vite dev server on http://localhost:5173
                                 #   /api/* is proxied to Flask on :5000
npm run build                    # production build to dist/
```

Open <http://localhost:5173>. Generate a forest, click a cell to set the
ignition point, then **Run fire** to animate, or run the GA / NSGA-II to find
cuts that reduce burn.

## Algorithms

- **Simulator (truth source).** BFS fire-spread, 8-neighbor adjacency, gap
  blocks. Returns per-step burn lists so the frontend can replay the spread.
- **Single-objective GA.** Tournament selection + uniform crossover + bit-flip
  mutation + elitism. Scalar fitness `w_s·survived − w_b·burned − w_c·cut`.
  Stops on `is_fit_enough` (target thresholds met), `ConvergenceTracker`
  (patience), or `max_generations`.
- **NSGA-II.** Fast non-dominated sort + crowding distance, binary tournament
  on (rank, crowding). Objectives are `(survived, −burned)` so both are
  maximize. Returns the Pareto front sorted by survival.

## What "fit enough" means

`fitness.is_fit_enough(report, config)` returns `True` iff a single individual
clears all three thresholds in `FitnessConfig`:

- `survival_rate ≥ min_survival_rate` (of trees that remain after the cut)
- `burn_rate ≤ max_burn_rate` (of original trees)
- `cut_rate ≤ max_cut_rate` (cap on how much you may chop)

`fitness.is_population_fit_enough(reports, config)` returns the index of the
first individual that clears them all, or `None`. The GA uses these to stop
early when a satisfactory solution is found.

## References

See [`doc.md`](./doc.md) — Svelte/Flask docs, Veritasium fire-spread video and
simulator, the SOC-EA Springer paper. The Drossel–Schwabl model behind
Veritasium's simulator influenced the UI controls; the propagation model here
is deterministic per the project spec.

## License

MIT — see [`LICENSE`](./LICENSE).
