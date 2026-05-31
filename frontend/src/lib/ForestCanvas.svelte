<script>
  import { afterUpdate, createEventDispatcher } from 'svelte';

  /** Tree grid: 2D array of 0/1. */
  export let grid = [];
  /** Cut mask: optional 2D array of 0/1 (1 = cut). */
  export let cutMask = null;
  /** Burned cells: optional Set of "r,c" strings showing what burned. */
  export let burned = new Set();
  /** Currently-igniting cells (animated frame). */
  export let burning = new Set();
  /** Ignition point — [r, c] or null. */
  export let ignitionPoint = null;
  /** Heatmap: optional 2D array of per-cell ignition weights (baseline = 1). */
  export let heatmap = null;
  /** Cell size in pixels. */
  export let cellSize = 12;
  /** Whether clicks set the ignition point. */
  export let allowClickToIgnite = true;

  const dispatch = createEventDispatcher();

  let canvas;
  let rows = 0, cols = 0;

  $: rows = grid.length;
  $: cols = rows ? grid[0].length : 0;
  $: width = cols * cellSize;
  $: height = rows * cellSize;

  // Redraw after every Svelte update — covers prop changes (grid, cutMask,
  // burned, burning, ignitionPoint, cellSize) and the canvas resize that
  // happens when width/height attrs change (which clears the canvas).
  afterUpdate(() => { if (canvas) draw(); });

  function colorFor(r, c) {
    const isTree = grid[r][c] === 1;
    const key = `${r},${c}`;
    if (burning.has(key)) return '#f97316';   // active fire frame
    if (burned.has(key)) return '#3f2014';    // burned
    if (cutMask && cutMask[r][c] === 1 && isTree) return '#92400e';  // cut tree
    if (isTree) return '#22c55e';             // alive tree
    return '#1a2028';                          // empty
  }

  function draw() {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        ctx.fillStyle = colorFor(r, c);
        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
      }
    }
    // Heatmap overlay — translucent red wherever weights exceed baseline.
    // Drawn between cells and ignition marker so the marker stays on top.
    if (heatmap && heatmap.length === rows && heatmap[0]?.length === cols) {
      let maxExcess = 0;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const excess = heatmap[r][c] - 1.0;  // baseline = 1.0
          if (excess > maxExcess) maxExcess = excess;
        }
      }
      if (maxExcess > 0) {
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const excess = heatmap[r][c] - 1.0;
            if (excess > 0) {
              const alpha = Math.min(0.55, 0.55 * excess / maxExcess);
              ctx.fillStyle = `rgba(220, 38, 38, ${alpha})`;
              ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
            }
          }
        }
      }
    }
    // Draw ignition marker.
    if (ignitionPoint) {
      const [ir, ic] = ignitionPoint;
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 2;
      ctx.strokeRect(ic * cellSize + 1, ir * cellSize + 1, cellSize - 2, cellSize - 2);
    }
  }

  function handleClick(event) {
    if (!allowClickToIgnite || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const c = Math.floor(x / cellSize);
    const r = Math.floor(y / cellSize);
    if (r >= 0 && r < rows && c >= 0 && c < cols) {
      dispatch('select', { r, c });
    }
  }

</script>

<canvas
  bind:this={canvas}
  width={width}
  height={height}
  on:click={handleClick}
  style="cursor: {allowClickToIgnite ? 'crosshair' : 'default'};"
></canvas>

<style>
  canvas {
    display: block;
    background: #0a0f14;
    border: 1px solid var(--border);
    border-radius: 0.25rem;
    image-rendering: pixelated;
  }
</style>
