<script>
  /**
   * Generic evolution chart.
   *
   * `series` is a list of normalized stats per generation:
   *   [{ generation, survival_rate, burn_rate, fitness? }, ...]
   *
   * Survival and burn rates are 0..1 and drawn on the left axis (%).
   * Fitness (optional, GA only) is drawn on a secondary scale, normalized to
   * its own min/max so it shares the chart without dominating it.
   */
  export let series = [];
  export let title = 'Evolution';

  const width = 380, height = 220, padL = 44, padR = 44, padT = 28, padB = 36;

  $: maxGen = series.length ? Math.max(1, series[series.length - 1].generation) : 1;
  $: hasFitness = series.some(s => typeof s.fitness === 'number');
  $: fitnesses = series.map(s => s.fitness).filter(v => typeof v === 'number');
  $: fitMin = fitnesses.length ? Math.min(...fitnesses) : 0;
  $: fitMax = fitnesses.length ? Math.max(...fitnesses) : 1;
  $: fitRange = fitMax - fitMin || 1;

  function x(g) {
    return padL + (g / maxGen) * (width - padL - padR);
  }
  // Left axis: 0..1 (rate)
  function yRate(r) {
    return height - padB - r * (height - padT - padB);
  }
  // Right axis: fitness normalized to its own range
  function yFit(f) {
    return height - padB - ((f - fitMin) / fitRange) * (height - padT - padB);
  }

  $: survivalPath = series.map(s => `${x(s.generation)},${yRate(s.survival_rate)}`).join(' ');
  $: burnPath = series.map(s => `${x(s.generation)},${yRate(s.burn_rate)}`).join(' ');
  $: fitnessPath = hasFitness
    ? series.filter(s => typeof s.fitness === 'number')
        .map(s => `${x(s.generation)},${yFit(s.fitness)}`)
        .join(' ')
    : '';

  const rateTicks = [0, 0.25, 0.5, 0.75, 1.0];
</script>

<div class="chart">
  <svg width={width} height={height} aria-label="Evolution per generation: survival % and burn %">
    <text x={width / 2} y={16} text-anchor="middle" fill="#cbd5e1" font-size="12" font-weight="600">
      {title}
    </text>

    <!-- Horizontal grid + left % ticks -->
    {#each rateTicks as r}
      <line x1={padL} y1={yRate(r)} x2={width - padR} y2={yRate(r)}
            stroke="#1f2937" stroke-dasharray="2 3" />
      <text x={padL - 6} y={yRate(r) + 3} text-anchor="end" fill="#64748b" font-size="10">
        {Math.round(r * 100)}%
      </text>
    {/each}

    <!-- Axes -->
    <line x1={padL} y1={height - padB} x2={width - padR} y2={height - padB} stroke="#475569" />
    <line x1={padL} y1={padT} x2={padL} y2={height - padB} stroke="#475569" />
    {#if hasFitness}
      <line x1={width - padR} y1={padT} x2={width - padR} y2={height - padB} stroke="#475569" />
    {/if}

    <!-- X label -->
    <text x={(padL + width - padR) / 2} y={height - 6} text-anchor="middle" fill="#94a3b8" font-size="11">
      generation →
    </text>

    <!-- Lines -->
    {#if series.length > 1}
      <polyline points={burnPath} fill="none" stroke="#ef4444" stroke-width="2" />
      <polyline points={survivalPath} fill="none" stroke="#4ade80" stroke-width="2" />
      {#if hasFitness}
        <polyline points={fitnessPath} fill="none" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="3 3" />
      {/if}
    {/if}

    <!-- Right-axis fitness range labels -->
    {#if hasFitness}
      <text x={width - padR + 6} y={padT + 4} fill="#fbbf24" font-size="10">{fitMax.toFixed(0)}</text>
      <text x={width - padR + 6} y={height - padB} fill="#fbbf24" font-size="10">{fitMin.toFixed(0)}</text>
      <text x={width - padR + 6} y={(padT + height - padB) / 2} fill="#fbbf24" font-size="10">fit</text>
    {/if}

    <!-- Legend -->
    <g transform={`translate(${padL + 6}, ${padT + 6})`}>
      <rect x="-4" y="-4" width={hasFitness ? 132 : 102} height="44" fill="#0a0f14" opacity="0.7" rx="3" />
      <line x1="0" y1="4" x2="12" y2="4" stroke="#4ade80" stroke-width="2" />
      <text x="16" y="7" fill="#cbd5e1" font-size="10">survival %</text>
      <line x1="0" y1="20" x2="12" y2="20" stroke="#ef4444" stroke-width="2" />
      <text x="16" y="23" fill="#cbd5e1" font-size="10">burn %</text>
      {#if hasFitness}
        <line x1="0" y1="36" x2="12" y2="36" stroke="#fbbf24" stroke-width="1.5" stroke-dasharray="3 3" />
        <text x="16" y="39" fill="#cbd5e1" font-size="10">fitness (scaled)</text>
      {/if}
    </g>
  </svg>
</div>

<style>
  .chart svg {
    background: var(--panel-2);
    border-radius: 0.25rem;
    width: 100%;
    height: auto;
    max-width: 380px;
  }
</style>
