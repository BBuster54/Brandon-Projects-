import { it, expect, vi } from "vitest";
import {
  normalizeAir,
  resolveAir,
  createAirProvider,
  airURL,
} from "../js/data/providers/openMeteo.js";
import { CACHE_TTL } from "../js/state.js";
import citiesData from "../data/cities.json";
import sampleData from "../data/sample-metrics.json";
const cities = citiesData.cities,
  samples = Object.fromEntries(sampleData.cities.map((c) => [c.id, c.metrics]));
const now = Date.parse("2026-10-20T12:00:00Z");
const fixture = cities.map(() => ({
  current: {
    time: now / 1000,
    us_aqi: 42,
    pm2_5: 8,
    pm10: 18,
    ozone: 50,
    nitrogen_dioxide: 10,
    sulphur_dioxide: 2,
    carbon_monoxide: 150,
  },
  hourly: {
    time: [now / 1000 - 3600, now / 1000, now / 1000 + 3600],
    us_aqi: [40, 42, 43],
  },
}));
it("one request has all cities and specified current pollutants", () => {
  const u = new URL(airURL(cities));
  expect(u.searchParams.get("latitude").split(",")).toHaveLength(5);
  expect(u.searchParams.get("current")).toContain("sulphur_dioxide");
  expect(u.searchParams.has("apikey")).toBe(false);
});
it("normalizes units and excludes future hours", () => {
  const rows = normalizeAir(fixture, cities, now);
  expect(rows.nyc.metrics.aqi.kind).toBe("live");
  expect(rows.nyc.metrics.co.unit).toBe("µg/m³");
  expect(rows.nyc.trend).toHaveLength(2);
});
it("partial response keeps other cities cached, marked stale", () => {
  const previous = normalizeAir(fixture, cities, now);
  const current = { nyc: previous.nyc };
  const resolved = resolveAir(cities, current, previous, samples, now);
  expect(resolved.records.la.aqi.stale).toBe(true);
  expect(resolved.records.nyc.aqi.stale).toBe(false);
});
it("no current or cache gives sample, no pollutants invented", () => {
  const r = resolveAir(cities, {}, {}, samples, now);
  expect(r.records.nyc.aqi.kind).toBe("sample");
  expect(r.records.nyc.pm25.value).toBeNull();
});
it("uses TTL and does not bypass it on refresh", async () => {
  let t = now;
  const fetchImpl = vi.fn(async () => ({
    ok: true,
    json: async () => fixture,
  }));
  const p = createAirProvider(cities, samples, {
    fetchImpl,
    storage: null,
    now: () => t,
  });
  await p.load();
  await p.load();
  expect(fetchImpl).toHaveBeenCalledTimes(1);
  t += CACHE_TTL;
  await p.load();
  expect(fetchImpl).toHaveBeenCalledTimes(2);
});
it("network failure returns stale values before samples", async () => {
  const entry = {
    fetchedAt: now - CACHE_TTL - 1,
    byCity: normalizeAir(fixture, cities, now),
  };
  const storage = { getItem: () => JSON.stringify(entry), setItem: () => {} };
  const p = createAirProvider(cities, samples, {
    fetchImpl: async () => {
      throw new Error("offline");
    },
    storage,
    now: () => now,
  });
  const r = await p.load();
  expect(r.records.nyc.aqi.kind).toBe("live");
  expect(r.records.nyc.aqi.stale).toBe(true);
  expect(r.status).toContain("stale");
});
it("timeouts abort and fall back safely", async () => {
  const fetchImpl = (_url, { signal }) =>
    new Promise((_resolve, reject) =>
      signal.addEventListener("abort", () => reject(new Error("timeout"))),
    );
  const r = await createAirProvider(cities, samples, {
    fetchImpl,
    storage: null,
    now: () => now,
    timeoutMs: 5,
  }).load();
  expect(r.records.nyc.aqi.kind).toBe("sample");
  expect(r.status).toContain("No current reading");
});
it("coalesces simultaneous requests", async () => {
  const fetchImpl = vi.fn(async () => ({
    ok: true,
    json: async () => fixture,
  }));
  const p = createAirProvider(cities, samples, {
    fetchImpl,
    storage: null,
    now: () => now,
  });
  await Promise.all([p.load(), p.load()]);
  expect(fetchImpl).toHaveBeenCalledTimes(1);
});
