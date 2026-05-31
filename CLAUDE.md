# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

EV-Forest (EVolutionary Forest) is a Bio-Inspired AI university course project. It optimizes a **cut mask** over a 2D forest grid to minimize wildfire spread while maximizing surviving trees. Python (simulator + optimizers + Flask) + Svelte (canvas UI).

## Repo layout

- `backend/` — Python package `ev_forest` (managed with **uv**): `forest.py` (grid gen), `simulator.py` (fire BFS), `ignition.py` (ignition strategies), `fitness.py` (objectives + `is_fit_enough`), `chromosome.py` (genetic operators), `ga.py` (single-objective GA), `nsga2.py` (multi-objective NSGA-II), `server.py` (Flask).
- `backend/tests/` — pytest. The simulator tests are load-bearing — they pin the fire-spread invariants below.
- `frontend/` — Svelte 5 + Vite. Components in `src/lib/`: `ForestCanvas`, `Controls`, `ParetoChart`, `HistoryChart`. `vite.config.js` proxies `/api/*` to Flask on `:5000`.

## Common commands

Backend (run from `backend/`):

```bash
uv sync                   # install deps (creates backend/.venv)
uv run pytest tests/      # run all tests
uv run pytest tests/test_simulator.py::test_one_pixel_gap_blocks_spread -v   # single test
uv run ev-forest          # start Flask dev server on :5000
```

Frontend (run from `frontend/`):

```bash
npm install               # one-time
npm run dev               # Vite dev server on :5173 (proxies /api -> :5000)
npm run build             # production build
```

Both servers must be running for the UI to work — Vite proxies `/api/*` to Flask.

## Load-bearing invariants (don't break these — tests will catch you)

- Fire propagates via **8-neighbor adjacency** (diagonals included).
- A single empty cell **blocks** spread — fire never jumps a gap.
- `Forest` grids use `np.int8` with `0=empty, 1=tree`; the simulator extends to `2=burned` in `SimulationResult.final_grid`.
- The baseline `layout="baseline"` produces trees every 2 cells — each tree's 3×3 neighborhood is empty, so the layout is fire-safe by construction. Tests assert this.

## Architecture notes

- **Simulator is the single source of truth.** Both `ga.py` and `nsga2.py` evaluate via `fitness.evaluate`, which calls `ignition.expected_burn`, which calls `simulator.simulate_fire`. Don't add a parallel spread implementation; extend the one in `simulator.py`.
- **Two views of fitness, one report.** `FitnessReport` carries both a scalar (`scalar_fitness`, used by GA) and a tuple (`objectives = (survived, -burned)`, used by NSGA-II — both framed as maximize so sorts don't need sign juggling).
- **Ignition strategy is a modeling decision, not an optimizer detail.** `fitness.FitnessConfig.ignition_strategy` is one of `"fixed" | "random" | "worst_case"` — changing it reshapes the fitness landscape. Pick explicitly; never default silently for the user.
- **"Fit enough" stopping.** `fitness.is_fit_enough` returns True iff `survival_rate ≥ min`, `burn_rate ≤ max`, and `cut_rate ≤ max` — all three. The GA stops early when any individual passes. `ConvergenceTracker` handles the other stop condition (no improvement for `patience` gens).
- **Per-step animation.** `SimulationResult.burned_per_step` is a list-of-lists of `(r, c)` ignited at each BFS round. The frontend animates by walking this list with a `setTimeout`.

## Frontend specifics

- Svelte 5 with the new `mount()` API in `main.js` and event dispatchers in components (no runes mode).
- All HTTP calls go through `src/lib/api.js`, never inline in components.
- The forest grid is rendered to `<canvas>` with `image-rendering: pixelated` — keep cell size an integer or rendering looks blurry.

## References

See the References section in `doc.md` (Svelte/Flask docs, Veritasium fire-spread video and simulator, SOC-EA Springer paper). Kept there to avoid drift.

## Working notes

- `doc.md`'s TO DO list is written in Italian — leave it in Italian when editing.
- `uv.lock` and `package-lock.json` should be committed for reproducibility (matches `.gitignore`).
- The `backend/.venv` is created on the first `uv sync` and is gitignored.
