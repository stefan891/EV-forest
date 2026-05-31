<script>
  import { createEventDispatcher } from 'svelte';

  export let params;
  export let busy = false;

  const dispatch = createEventDispatcher();
  function emit(name) { dispatch(name); }
</script>

<div class="controls">
  <section class="panel">
    <h2>Forest</h2>
    <div class="row">
      <label class="field"><span>Rows</span>
        <input type="number" min="4" max="120" bind:value={params.rows} />
      </label>
      <label class="field"><span>Cols</span>
        <input type="number" min="4" max="120" bind:value={params.cols} />
      </label>
      <label class="field"><span>Layout</span>
        <select bind:value={params.layout}>
          <option value="dense">Dense (all trees)</option>
          <option value="random">Random</option>
          <option value="checkerboard">Checkerboard</option>
          <option value="baseline">Baseline (every 2nd)</option>
        </select>
      </label>
      <label class="field"><span>Density</span>
        <input type="number" step="0.05" min="0" max="1" bind:value={params.density} />
      </label>
      <label class="field"><span>Seed</span>
        <input type="number" bind:value={params.seed} />
      </label>
    </div>
    <div class="row" style="margin-top: 0.75rem;">
      <button class="primary" disabled={busy} on:click={() => emit('generate')}>Generate forest</button>
      <button disabled={busy} on:click={() => emit('simulate')}>Run fire</button>
    </div>
  </section>

  <section class="panel">
    <h2>Fitness thresholds <span class="muted">(is_fit_enough)</span></h2>
    <div class="row">
      <label class="field"><span>Max burn</span>
        <input type="number" step="0.01" min="0" max="1" bind:value={params.max_burn_rate} />
      </label>
      
      <label class="field"><span>Max cut</span>
        <input type="number" step="0.05" min="0" max="1" bind:value={params.max_cut_rate} />
      </label>
    </div>
  </section>

  <section class="panel">
    <h2>Optimization</h2>
    <div class="row">
      <label class="field"><span>Population</span>
        <input type="number" min="10" max="500" bind:value={params.population_size} />
      </label>
      <label class="field"><span>Generations</span>
        <input type="number" min="5" max="500" bind:value={params.max_generations} />
      </label>
      <label class="field"><span>Mutation</span>
        <input type="number" step="0.005" min="0" max="1" bind:value={params.mutation_rate} />
      </label>
      <label class="field"><span>Crossover</span>
        <input type="number" step="0.05" min="0" max="1" bind:value={params.crossover_rate} />
      </label>
      <label class="field"><span>Init cut %</span>
        <input type="number" step="0.05" min="0" max="1" bind:value={params.initial_cut_probability} />
      </label>
      <label class="field"><span>Optimizer seed</span>
        <input type="number" bind:value={params.optimizer_seed} />
      </label>
      <label class="field"><span>Ignition strategy</span>
        <select bind:value={params.ignition_strategy}>
          <option value="random">Random (sample best)</option>
          <option value="worst_case">Worst-case (exhaustive search)</option>
        </select>
      </label>
      <label class="field"><span>Ignition samples</span>
        <input type="number" min="1" max="64" bind:value={params.ignition_samples} />
      </label>
      <label class="field"><span>GA selection</span>
        <select bind:value={params.selection_strategy}>
          <option value="tournament">Tournament</option>
          <option value="rank_based">Rank-based</option>
        </select>
      </label>
      <label class="field"><span>Tournament size</span>
        <input type="number" min="2" max="20" bind:value={params.tournament_size} />
      </label>
      <label class="field"><span>Patch size (GA)</span>
        <input type="number" min="1" max="10" bind:value={params.patch_size} />
      </label>
    </div>
    <p class="hint">
      Use <strong>random</strong> for faster sampling-based robustness check.
      Use <strong>worst-case</strong> for true worst-case ignition point (exhaustive search, slower).
    </p>
    <div class="row" style="margin-top: 0.75rem;">
      <button class="primary" disabled={busy} on:click={() => emit('runGA')}>Run GA (single-objective)</button>
      <button class="primary" disabled={busy} on:click={() => emit('runNSGA2')}>Run NSGA-II (Pareto front)</button>
    </div>
  </section>
</div>

<style>
  .controls {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  .field > span {
    font-size: 0.8rem;
    color: var(--text-dim);
  }
  .field > input, .field > select {
    width: 6rem;
  }
  .hint {
    margin: 0.5rem 0 0;
    font-size: 0.75rem;
    color: var(--text-dim);
    line-height: 1.4;
  }
</style>
