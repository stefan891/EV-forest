<script>
  import { createEventDispatcher } from 'svelte';

  /** Reports on the Pareto front — clickable, highlighted in green. */
  export let frontPoints = [];
  /** All final-population reports — shown as a gray cloud behind the front. */
  export let population = [];
  /** No-cut baseline report — shown as a red marker so the user sees the win. */
  export let baseline = null;
  /** Index of the currently selected front point. */
  export let selectedIndex = -1;

  const dispatch = createEventDispatcher();

  const width = 380, height = 280, padL = 50, padR = 14, padT = 28, padB = 46;

  // Build a combined value range so every point fits on the same axes.
  $: allReports = [
    ...population,
    ...frontPoints,
    ...(baseline ? [baseline] : []),
  ];
  $: maxCut = allReports.length ? Math.max(...allReports.map(r => r.trees_cut), 1) : 1;
  $: maxSurvived = allReports.length ? Math.max(...allReports.map(r => r.trees_survived), 1) : 1;
  $: original = allReports.length ? allReports[0].trees_original : 1;

  function x(p) {
    return padL + (p.trees_cut / maxCut) * (width - padL - padR);
  }
  function y(p) {
    return height - padB - (p.trees_survived / maxSurvived) * (height - padT - padB);
  }

  // Tick marks at 0, 25, 50, 75, 100% of original tree count.
  $: xTicks = [0, 0.25, 0.5, 0.75, 1.0]
    .map(f => f * original)
    .filter(v => v <= maxCut * 1.01);
  $: yTicks = [0, 0.25, 0.5, 0.75, 1.0]
    .map(f => f * original)
    .filter(v => v <= maxSurvived * 1.01);

  function tickX(v) {
    return padL + (v / maxCut) * (width - padL - padR);
  }
  function tickY(v) {
    return height - padB - (v / maxSurvived) * (height - padT - padB);
  }
</script>

<div class="chart">
  <svg width={width} height={height} aria-label="Pareto front: trees burned vs trees survived">
    <!-- Title -->
    <text x={width / 2} y={16} text-anchor="middle" fill="#cbd5e1" font-size="12" font-weight="600">
      Trade-off: fewer cuts ↔ more survived
    </text>

    <!-- Grid lines -->
    {#each xTicks as v}
      <line x1={tickX(v)} y1={padT} x2={tickX(v)} y2={height - padB}
            stroke="#1f2937" stroke-dasharray="2 3" />
    {/each}
    {#each yTicks as v}
      <line x1={padL} y1={tickY(v)} x2={width - padR} y2={tickY(v)}
            stroke="#1f2937" stroke-dasharray="2 3" />
    {/each}

    <!-- Axes -->
    <line x1={padL} y1={height - padB} x2={width - padR} y2={height - padB} stroke="#475569" />
    <line x1={padL} y1={padT} x2={padL} y2={height - padB} stroke="#475569" />

    <!-- X tick labels (% of original trees) -->
    {#each xTicks as v}
      <text x={tickX(v)} y={height - padB + 14} text-anchor="middle" fill="#64748b" font-size="10">
        {Math.round((v / original) * 100)}%
      </text>
    {/each}
    <!-- Y tick labels -->
    {#each yTicks as v}
      <text x={padL - 6} y={tickY(v) + 3} text-anchor="end" fill="#64748b" font-size="10">
        {Math.round((v / original) * 100)}%
      </text>
    {/each}

    <!-- Axis labels -->
    <text x={(padL + width - padR) / 2} y={height - 8} text-anchor="middle" fill="#94a3b8" font-size="11">
      trees cut (% of original) — lower is better →
    </text>
    <text x={14} y={(padT + height - padB) / 2} fill="#94a3b8" font-size="11"
          transform={`rotate(-90, 14, ${(padT + height - padB) / 2})`} text-anchor="middle">
      trees survived (% of original) →
    </text>

    <!-- Population cloud (gray dots in the background) -->
    {#each population as p}
      <circle cx={x(p)} cy={y(p)} r="2.5" fill="#475569" opacity="0.55" />
    {/each}

    <!-- Pareto front connector -->
    {#if frontPoints.length > 1}
      <polyline
        points={frontPoints.map(p => `${x(p)},${y(p)}`).join(' ')}
        fill="none"
        stroke="#4ade80"
        stroke-width="1.5"
        stroke-dasharray="4 3"
        opacity="0.7"
      />
    {/if}

    <!-- Pareto front points -->
    {#each frontPoints as p, i}
      <circle
        role="button"
        tabindex="0"
        aria-label={`Pareto point ${i + 1}: ${p.trees_burned} burned, ${p.trees_survived} survived, ${p.trees_cut} cut`}
        cx={x(p)} cy={y(p)}
        r={i === selectedIndex ? 7 : 5}
        fill={i === selectedIndex ? '#fbbf24' : '#4ade80'}
        stroke="#0a0f14" stroke-width="1.5"
        style="cursor: pointer;"
        on:click={() => dispatch('select', i)}
        on:keydown={(e) => (e.key === 'Enter' || e.key === ' ') && dispatch('select', i)}
      >
        <title>burned: {p.trees_burned} · survived: {p.trees_survived} · cut: {p.trees_cut}</title>
      </circle>
    {/each}

    <!-- No-cut baseline marker (red X) -->
    {#if baseline}
      {@const bx = x(baseline)}
      {@const by = y(baseline)}
      <g>
        <line x1={bx - 6} y1={by - 6} x2={bx + 6} y2={by + 6} stroke="#ef4444" stroke-width="2" />
        <line x1={bx - 6} y1={by + 6} x2={bx + 6} y2={by - 6} stroke="#ef4444" stroke-width="2" />
        <title>baseline (no cut): burned {baseline.trees_burned}, survived {baseline.trees_survived}</title>
      </g>
    {/if}

    <!-- Legend -->
    <g transform={`translate(${width - padR - 110}, ${padT + 6})`}>
      <rect x="-6" y="-6" width="116" height="58" fill="#0a0f14" opacity="0.7" rx="3" />
      <circle cx="4" cy="4" r="4" fill="#4ade80" stroke="#0a0f14" stroke-width="1" />
      <text x="14" y="7" fill="#cbd5e1" font-size="10">Pareto front</text>
      <circle cx="4" cy="20" r="2.5" fill="#475569" />
      <text x="14" y="23" fill="#cbd5e1" font-size="10">population</text>
      <line x1="-1" y1="36" x2="9" y2="36" stroke="#ef4444" stroke-width="2" transform="rotate(45, 4, 36)" />
      <line x1="-1" y1="36" x2="9" y2="36" stroke="#ef4444" stroke-width="2" transform="rotate(-45, 4, 36)" />
      <text x="14" y="39" fill="#cbd5e1" font-size="10">no-cut baseline</text>
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
