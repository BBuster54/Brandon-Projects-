import { metricDefinitions } from "./data/normalize.js";
export const defaultState = {
  view: "overview",
  city: "nyc",
  metric: "aqi",
  compare: "la",
  category: "all",
  sort: "desc",
};
export function parseState(search, cities) {
  const p = new URLSearchParams(search),
    ids = cities.map((c) => c.id),
    state = { ...defaultState };
  const choices = {
    view: ["overview", "environment", "transportation", "housing"],
    city: ids,
    metric: Object.keys(metricDefinitions),
    compare: ids,
    category: [
      "all",
      "good",
      "moderate",
      "sensitive",
      "unhealthy",
      "very-unhealthy",
      "hazardous",
      "unavailable",
    ],
    sort: ["asc", "desc"],
  };
  for (const [k, v] of Object.entries(choices))
    if (v.includes(p.get(k))) state[k] = p.get(k);
  return state;
}
export function stateQuery(state) {
  return new URLSearchParams(Object.entries(state)).toString();
}
export function syncURL(state) {
  try {
    history.replaceState(null, "", location.pathname + "?" + stateQuery(state));
  } catch {
    /* Embedded browsing context may block URL changes. */
  }
}
export function storageRead(storage, key, fallback = null) {
  try {
    return JSON.parse(storage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}
export function storageWrite(storage, key, value) {
  try {
    storage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}
export function safeStorage(name) {
  try {
    return globalThis[name];
  } catch {
    return null;
  }
}
export const CACHE_TTL = 15 * 60 * 1000;
export function cacheFresh(entry, now = Date.now()) {
  return Boolean(
    entry &&
      Number.isFinite(entry.fetchedAt) &&
      now >= entry.fetchedAt &&
      now - entry.fetchedAt < CACHE_TTL,
  );
}
