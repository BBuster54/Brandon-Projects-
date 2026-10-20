import { numeric, missingMetric, selectFallback } from "../normalize.js";
import {
  safeStorage,
  storageRead,
  storageWrite,
  cacheFresh,
} from "../../state.js";
export const CACHE_KEY = "urban-air-v2";
export const airFields = {
  aqi: "us_aqi",
  pm25: "pm2_5",
  pm10: "pm10",
  ozone: "ozone",
  no2: "nitrogen_dioxide",
  so2: "sulphur_dioxide",
  co: "carbon_monoxide",
};
export function airURL(cities) {
  const q = new URLSearchParams({
    latitude: cities.map((c) => c.lat).join(","),
    longitude: cities.map((c) => c.lon).join(","),
    current: Object.values(airFields).join(","),
    hourly: "us_aqi",
    past_hours: "24",
    forecast_hours: "1",
    timezone: "GMT",
    timeformat: "unixtime",
    domains: "cams_global",
  });
  return "https://air-quality-api.open-meteo.com/v1/air-quality?" + q;
}
export function normalizeAir(payload, cities, now = Date.now()) {
  const responses = Array.isArray(payload) ? payload : [payload],
    records = {};
  for (let i = 0; i < cities.length; i++) {
    const raw = responses[i],
      current = raw?.current;
    if (!current || numeric(current.time) === null) continue;
    const asOf = new Date(current.time * 1000).toISOString(),
      metrics = {};
    for (const [key, field] of Object.entries(airFields)) {
      const value = numeric(current[field]);
      if (value === null) continue;
      metrics[key] = {
        value,
        unit: key === "aqi" ? "US AQI" : "µg/m³",
        source: "Modeled — CAMS global via Open-Meteo",
        sourceUrl: airURL([cities[i]]),
        asOf,
        kind: "live",
        method:
          "Coarse CAMS global model (~45 km), not a local monitor. City-center grid values can differ from AirNow readings.",
        retrievedAt: new Date(now).toISOString(),
      };
    }
    const hours = raw.hourly?.time ?? [],
      values = raw.hourly?.us_aqi ?? [];
    const trend = hours
      .map((t, j) => ({
        time: typeof t === "number" ? new Date(t * 1000).toISOString() : null,
        value: numeric(values[j]),
      }))
      .filter((p) => p.time && p.value !== null && Date.parse(p.time) <= now)
      .slice(-24);
    records[cities[i].id] = { metrics, trend };
  }
  return records;
}
export function resolveAir(cities, current, cached, samples, now = Date.now()) {
  const output = {},
    trends = {};
  for (const c of cities) {
    output[c.id] = {};
    for (const key of Object.keys(airFields)) {
      const chosen = selectFallback(
        current[c.id]?.metrics?.[key],
        cached[c.id]?.metrics?.[key],
        key === "aqi" ? samples[c.id]?.aqi : null,
        now,
      );
      output[c.id][key] = chosen ?? missingMetric(key);
    }
    const active = current[c.id]?.metrics?.aqi ? current[c.id] : cached[c.id];
    trends[c.id] = active?.trend ?? [];
  }
  return { records: output, trends };
}
export function createAirProvider(
  cities,
  samples,
  {
    fetchImpl = globalThis.fetch,
    storage = safeStorage("sessionStorage"),
    now = () => Date.now(),
    timeoutMs = 8000,
  } = {},
) {
  let entry = storageRead(storage, CACHE_KEY, null),
    pending = null,
    lastAttempt = Number(entry?.lastAttempt) || 0;
  async function load() {
    if (pending) return pending;
    const time = now();
    if (cacheFresh(entry, time))
      return {
        ...resolveAir(cities, entry.byCity ?? {}, {}, samples, time),
        cached: true,
        status:
          "Cached modeled air quality · respects the 15-minute refresh window.",
      };
    if (time - lastAttempt < 60000)
      return {
        ...resolveAir(cities, {}, entry?.byCity ?? {}, samples, time),
        cached: true,
        status:
          "Refresh cooldown · showing last-good stale data or explicitly labeled sample fallback.",
      };
    lastAttempt = time;
    pending = (async () => {
      const controller = new AbortController(),
        timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const response = await fetchImpl(airURL(cities), {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error("Provider unavailable");
        const current = normalizeAir(await response.json(), cities, time);
        if (!Object.values(current).some((r) => r.metrics.aqi))
          throw new Error("No valid air quality");
        const resolved = resolveAir(
          cities,
          current,
          entry?.byCity ?? {},
          samples,
          time,
        );
        const byCity = {};
        for (const c of cities) {
          const merged = {
            ...Object.fromEntries(
              Object.entries(entry?.byCity?.[c.id]?.metrics ?? {}).map(
                ([k, m]) => [k, { ...m, stale: true }],
              ),
            ),
            ...(current[c.id]?.metrics ?? {}),
          };
          byCity[c.id] = {
            metrics: merged,
            trend: current[c.id]?.trend?.length
              ? current[c.id].trend
              : (entry?.byCity?.[c.id]?.trend ?? []),
          };
        }
        entry = { fetchedAt: time, lastAttempt: time, byCity };
        storageWrite(storage, CACHE_KEY, entry);
        return {
          ...resolved,
          cached: false,
          status:
            "Modeled air quality — CAMS via Open-Meteo. City-center estimates can differ from local monitors.",
        };
      } catch {
        if (entry) {
          entry = { ...entry, lastAttempt: time };
          storageWrite(storage, CACHE_KEY, entry);
        }
        const fallback = resolveAir(
          cities,
          {},
          entry?.byCity ?? {},
          samples,
          time,
        );
        const hasStale = Object.values(fallback.records).some(
          (r) => r.aqi.kind === "live",
        );
        return {
          ...fallback,
          cached: false,
          status: hasStale
            ? "Live update unavailable · showing last-good readings marked stale, as of their displayed local times."
            : "Live update unavailable · Sample fallback only. No current reading; sample markers are neutral and hatched.",
        };
      } finally {
        clearTimeout(timer);
        pending = null;
      }
    })();
    return pending;
  }
  return { load };
}
