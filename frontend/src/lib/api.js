// Thin fetch wrapper for the Flask backend. The /api prefix is proxied by Vite
// to http://127.0.0.1:5000 (see vite.config.js).

async function post(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${path} -> ${res.status}: ${text}`);
  }
  return res.json();
}

export function generateForest(params) {
  return post('/api/forest', params);
}

export function generateHeatmap(params) {
  return post('/api/heatmap', params);
}

export function simulate(params) {
  return post('/api/simulate', params);
}

export function optimizeGA(params) {
  return post('/api/optimize/ga', params);
}

export function optimizeNSGA2(params) {
  return post('/api/optimize/nsga2', params);
}
