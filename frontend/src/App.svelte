<script>
  import { onMount } from 'svelte';
  import { generateForest, generateHeatmap, simulate, optimizeGA, optimizeNSGA2 } from './lib/api.js';
  import ForestCanvas from './lib/ForestCanvas.svelte';
  import Controls from './lib/Controls.svelte';
  import ParetoChart from './lib/ParetoChart.svelte';
  import HistoryChart from './lib/HistoryChart.svelte';

  let params = {
    rows: 30,
    cols: 30,
    layout: 'dense',
    density: 1.0,
    seed: 0,
    n_hotspots: 3,
    heatmap_seed: 0,
    hotspot_strength: 6.0,
    max_burn_rate: 0.10,
    max_cut_rate: 0.30,
    population_size: 60,
    max_generations: 50,
    mutation_rate: 0.01,
    crossover_rate: 0.9,
    initial_cut_probability: 0.1,
    optimizer_seed: 1,
    ignition_strategy: 'random',
    ignition_samples: 12,
    selection_strategy: 'tournament',
    tournament_size: 3,
    patch_size: 2,
  };

  let grid = [];
  let cutMask = null;
  let heatmap = null;
  let ignitionPoint = null;
  let burnedSet = new Set();
  let burningSet = new Set();
  let simResult = null;
  let gaResult = null;
  let nsgaResult = null;
  let selectedFrontIdx = -1;
  let busy = false;
  let status = '';
  let mode = 'idle';  // 'idle' | 'ga' | 'nsga2'

  onMount(() => doGenerate());

  function emptyMask() {
    return grid.map(row => row.map(() => 0));
  }

  function key(r, c) { return `${r},${c}`; }

  async function doGenerate() {
    busy = true;
    status = 'generating forest…';
    try {
      const r = await generateForest({
        rows: params.rows, cols: params.cols, layout: params.layout,
        density: params.density, seed: params.seed,
        n_hotspots: params.n_hotspots,
        heatmap_seed: params.heatmap_seed,
        hotspot_strength: params.hotspot_strength,
      });
      grid = r.grid;
      heatmap = r.heatmap;
      cutMask = emptyMask();
      burnedSet = new Set();
      burningSet = new Set();
      ignitionPoint = [Math.floor(params.rows / 2), Math.floor(params.cols / 2)];
      simResult = null;
      gaResult = null;
      nsgaResult = null;
      selectedFrontIdx = -1;
      mode = 'idle';
      status = `forest: ${r.tree_count} trees, layout=${r.layout}, hotspots=${params.n_hotspots}`;
    } catch (e) {
      status = `error: ${e.message}`;
    } finally {
      busy = false;
    }
  }

  async function doRegenHeatmap() {
    if (!grid.length) return;
    busy = true;
    status = 'regenerating hot zones…';
    try {
      const r = await generateHeatmap({
        rows: grid.length, cols: grid[0].length,
        n_hotspots: params.n_hotspots,
        heatmap_seed: params.heatmap_seed,
        hotspot_strength: params.hotspot_strength,
      });
      heatmap = r.heatmap;
      status = `hot zones updated: ${params.n_hotspots} hotspot(s), seed=${params.heatmap_seed}`;
    } catch (e) {
      status = `error: ${e.message}`;
    } finally {
      busy = false;
    }
  }

  async function doSimulate() {
    if (!grid.length || !ignitionPoint) return;
    busy = true;
    status = 'simulating fire…';
    try {
      const r = await simulate({
        grid,
        cut_mask: cutMask,
        ignition_point: ignitionPoint,
      });
      simResult = r;
      await animateBurn(r.burned_per_step);
      status = `burned ${r.burned}/${r.total_trees} in ${r.steps} steps`;
    } catch (e) {
      status = `error: ${e.message}`;
    } finally {
      busy = false;
    }
  }

  async function animateBurn(stepsList) {
    burnedSet = new Set();
    burningSet = new Set();
    for (const step of stepsList) {
      burningSet = new Set(step.map(([r, c]) => key(r, c)));
      await sleep(120);
      burningSet.forEach(k => burnedSet.add(k));
      burningSet = new Set();
      burnedSet = new Set(burnedSet);  // trigger reactivity
    }
  }

  function sleep(ms) {
    return new Promise(res => setTimeout(res, ms));
  }

  function onCanvasSelect(e) {
    ignitionPoint = [e.detail.r, e.detail.c];
    burnedSet = new Set();
    burningSet = new Set();
  }

  async function doRandomSpark() {
    if (!grid.length) return;
    // Build a weighted candidate list. If no heatmap, weights default to 1
    // (uniform). Only living, uncut trees are eligible to ignite.
    const candidates = [];
    const weights = [];
    let totalWeight = 0;
    for (let r = 0; r < grid.length; r++) {
      for (let c = 0; c < grid[0].length; c++) {
        if (grid[r][c] === 1 && (!cutMask || cutMask[r][c] !== 1)) {
          const w = heatmap ? Math.max(0, heatmap[r][c]) : 1;
          candidates.push([r, c]);
          weights.push(w);
          totalWeight += w;
        }
      }
    }
    if (!candidates.length || totalWeight <= 0) {
      status = 'no tree cells available to ignite';
      return;
    }
    let pick = Math.random() * totalWeight;
    let chosen = candidates[candidates.length - 1];
    for (let i = 0; i < candidates.length; i++) {
      pick -= weights[i];
      if (pick <= 0) { chosen = candidates[i]; break; }
    }
    ignitionPoint = chosen;
    burnedSet = new Set();
    burningSet = new Set();
    await doSimulate();
  }

  function makeOptimizerBody() {
    return {
      grid,
      heatmap,
      population_size: params.population_size,
      max_generations: params.max_generations,
      selection_strategy: params.selection_strategy,
      tournament_size: params.tournament_size,
      mutation_rate: params.mutation_rate,
      crossover_rate: params.crossover_rate,
      initial_cut_probability: params.initial_cut_probability,
      patch_size: params.patch_size,
      seed: params.optimizer_seed,
      ignition_point: ignitionPoint,
      ignition_strategy: params.ignition_strategy,
      ignition_samples: params.ignition_samples,
      ignition_seed: params.optimizer_seed,
      max_burn_rate: params.max_burn_rate,
      max_cut_rate: params.max_cut_rate,
    };
  }

  async function doRunGA() {
    if (!grid.length) return;
    busy = true;
    status = 'running GA…';
    mode = 'ga';
    nsgaResult = null;
    selectedFrontIdx = -1;
    try {
      const r = await optimizeGA(makeOptimizerBody());
      gaResult = r;
      cutMask = r.best_cut_mask;
      burnedSet = new Set();
      burningSet = new Set();
      const rep = r.best_report;
      status = `GA → ${r.stopped_reason} after ${r.generations_run} gens · burned ${rep.trees_burned} · cut ${rep.trees_cut} · survived ${rep.trees_survived} (running fire…)`;
    } catch (e) {
      status = `error: ${e.message}`;
      busy = false;
      return;
    }
    busy = false;
    // Auto-play the fire on the optimized cut so the user sees the result.
    await doSimulate();
  }

  async function doRunNSGA2() {
    if (!grid.length) return;
    busy = true;
    status = 'running NSGA-II…';
    mode = 'nsga2';
    gaResult = null;
    try {
      const r = await optimizeNSGA2(makeOptimizerBody());
      nsgaResult = r;
      status = `NSGA-II → ${r.pareto_front.length} points on Pareto front (${r.generations_run} gens)`;
      // Auto-select the "middle" point (best balanced trade-off) and play the fire.
      if (r.pareto_front.length) {
        const midIdx = Math.floor(r.pareto_front.length / 2);
        selectedFrontIdx = midIdx;
        cutMask = r.pareto_front[midIdx].cut_mask;
        burnedSet = new Set();
        burningSet = new Set();
        busy = false;
        await doSimulate();
        return;
      }
    } catch (e) {
      status = `error: ${e.message}`;
    } finally {
      busy = false;
    }
  }

  async function applyFrontPoint(i) {
    if (!nsgaResult || i < 0 || i >= nsgaResult.pareto_front.length) return;
    selectedFrontIdx = i;
    cutMask = nsgaResult.pareto_front[i].cut_mask;
    burnedSet = new Set();
    burningSet = new Set();
    // Auto-play the fire on the picked cut so the trade-off is visible.
    await doSimulate();
  }

  /** Normalize GA history into the shape HistoryChart expects. */
  function gaSeries(gaR) {
    return gaR.history.map(h => ({
      generation: h.generation,
      survival_rate: h.best_report.trees_original
        ? h.best_report.trees_survived / h.best_report.trees_original
        : 0,
      burn_rate: h.best_report.trees_original
        ? h.best_report.trees_burned / h.best_report.trees_original
        : 0,
      fitness: h.best_fitness,
    }));
  }

  /** Normalize NSGA-II history: pick the best (max survived, min burned) on the front each gen. */
  function nsgaSeries(nR) {
    const original = nR.baseline?.trees_original
      || (nR.population && nR.population[0]?.trees_original)
      || 1;
    return nR.history.map(h => {
      // front_objectives is [[survived, -burned], ...]
      const survived = h.front_objectives.length
        ? Math.max(...h.front_objectives.map(o => o[0]))
        : 0;
      const burned = h.front_objectives.length
        ? Math.min(...h.front_objectives.map(o => -o[1]))
        : 0;
      return {
        generation: h.generation,
        survival_rate: survived / original,
        burn_rate: burned / original,
      };
    });
  }
</script>

<main>
  <header>
    <h1>EV-Forest <span class="muted">— Bio-Inspired AI · optimize cuts vs fire spread</span></h1>
    <div class="status">{status}</div>
  </header>

  <div class="layout">
    <aside>
      <Controls
        bind:params
        {busy}
        on:generate={doGenerate}
        on:regenHeatmap={doRegenHeatmap}
        on:simulate={doSimulate}
        on:randomSpark={doRandomSpark}
        on:runGA={doRunGA}
        on:runNSGA2={doRunNSGA2}
      />
    </aside>

    <section class="canvas-area panel">
      <div class="canvas-header">
        <h2>Forest grid</h2>
        <div class="legend">
          <span><i style="background:#22c55e"></i> tree</span>
          <span><i style="background:#92400e"></i> cut</span>
          <span><i style="background:#f97316"></i> burning</span>
          <span><i style="background:#3f2014"></i> burned</span>
          <span><i style="background:transparent;border:1px solid #fbbf24"></i> ignition</span>
        </div>
      </div>
      {#if grid.length}
        <ForestCanvas
          {grid}
          {cutMask}
          {heatmap}
          burned={burnedSet}
          burning={burningSet}
          {ignitionPoint}
          cellSize={Math.max(6, Math.min(20, Math.floor(560 / Math.max(grid.length, grid[0]?.length || 1))))}
          on:select={onCanvasSelect}
        />
        <p class="muted">Click any cell to set the ignition point, then press <strong>Run fire</strong>.</p>
      {:else}
        <p class="muted">No forest generated yet.</p>
      {/if}
    </section>

    <aside class="results">
      {#if mode === 'ga' && gaResult}
        <section class="panel">
          <h2>GA learning curve</h2>
          <HistoryChart series={gaSeries(gaResult)} title="GA: best per generation" />
          <div class="kv">
            <div><span class="muted">stopped:</span> {gaResult.stopped_reason}</div>
            <div><span class="muted">scalar fitness:</span> {gaResult.best_report.scalar_fitness.toFixed(2)}</div>
            <div><span class="muted">survived:</span> {gaResult.best_report.trees_survived} / {gaResult.best_report.trees_original}</div>
            <div><span class="muted">burned:</span> {gaResult.best_report.trees_burned}</div>
            <div><span class="muted">worst-case burned:</span> {gaResult.worst_case_burned}</div>
            <div><span class="muted">cut:</span> {gaResult.best_report.trees_cut}</div>
          </div>
          {#if gaResult.baseline}
            <p class="muted compare">
              vs. no-cut baseline: <strong>{gaResult.baseline.trees_burned}</strong> would burn,
              <strong>{gaResult.baseline.trees_survived}</strong> survive →
              GA saved <strong>{gaResult.baseline.trees_burned - gaResult.best_report.trees_burned}</strong> trees.
            </p>
          {/if}
        </section>
      {/if}

      {#if mode === 'nsga2' && nsgaResult}
        <section class="panel">
          <h2>NSGA-II Pareto front</h2>
          <ParetoChart
            frontPoints={nsgaResult.pareto_front.map(p => p.report)}
            population={nsgaResult.population || []}
            baseline={nsgaResult.baseline}
            selectedIndex={selectedFrontIdx}
            on:select={(e) => applyFrontPoint(e.detail)}
          />
          <p class="muted">Click a point to view that cut-mask on the grid.</p>
          {#if selectedFrontIdx >= 0}
            {@const rep = nsgaResult.pareto_front[selectedFrontIdx].report}
            {@const worstBurned = nsgaResult.pareto_front[selectedFrontIdx].worst_case_burned}
            <div class="kv">
              <div><span class="muted">survived:</span> {rep.trees_survived} / {rep.trees_original}</div>
              <div><span class="muted">burned:</span> {rep.trees_burned}</div>
              <div><span class="muted">worst-case burned:</span> {worstBurned}</div>
              <div><span class="muted">cut:</span> {rep.trees_cut} ({(rep.cut_rate * 100).toFixed(1)}%)</div>
            </div>
          {/if}
        </section>
        <section class="panel">
          <h2>NSGA-II evolution</h2>
          <HistoryChart series={nsgaSeries(nsgaResult)} title="NSGA-II: front best per generation" />
          <p class="muted">
            Best survival % and lowest burn % found on the Pareto front at each generation.
          </p>
        </section>
      {/if}

      {#if mode === 'idle' && simResult}
        <section class="panel">
          <h2>Last simulation</h2>
          <div class="kv">
            <div><span class="muted">trees:</span> {simResult.total_trees}</div>
            <div><span class="muted">burned:</span> {simResult.burned}</div>
            <div><span class="muted">survived:</span> {simResult.survived}</div>
            <div><span class="muted">steps:</span> {simResult.steps}</div>
          </div>
        </section>
      {/if}
    </aside>
  </div>
</main>

<style>
  main {
    max-width: 1400px;
    margin: 0 auto;
    padding: 1rem;
  }
  header {
    margin-bottom: 1rem;
  }
  .status {
    color: var(--text-dim);
    font-size: 0.85rem;
    margin-top: 0.25rem;
    min-height: 1.25rem;
  }
  .layout {
    display: grid;
    grid-template-columns: minmax(260px, 320px) 1fr minmax(260px, 380px);
    gap: 1rem;
    align-items: start;
  }
  .canvas-area {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    align-items: center;
  }
  .canvas-header {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  .legend {
    display: flex;
    gap: 0.6rem;
    font-size: 0.75rem;
    color: var(--text-dim);
  }
  .legend i {
    display: inline-block;
    width: 0.7rem;
    height: 0.7rem;
    margin-right: 0.25rem;
    vertical-align: middle;
    border-radius: 2px;
  }
  .results {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .kv {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.25rem 0.75rem;
    font-size: 0.85rem;
    margin-top: 0.5rem;
  }

  @media (max-width: 1100px) {
    .layout {
      grid-template-columns: 1fr;
    }
  }
</style>
