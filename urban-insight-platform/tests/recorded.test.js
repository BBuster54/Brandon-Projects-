import { it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { normalizeAir } from "../js/data/providers/openMeteo.js";
const fixture = JSON.parse(
  readFileSync(new URL("fixtures/open-meteo-recorded.json", import.meta.url)),
);
const { cities } = JSON.parse(
  readFileSync(new URL("../data/cities.json", import.meta.url)),
);
it("normalizes a recorded multi-city Open-Meteo response without shifting city order", () => {
  const result = normalizeAir(fixture, cities);
  for (let i = 0; i < cities.length; i++) {
    const r = result[cities[i].id];
    expect(r.metrics.aqi.value).toBe(fixture[i].current.us_aqi);
    expect(r.metrics.co.unit).toBe("µg/m³");
    expect(r.trend.length).toBeLessThanOrEqual(24);
    expect(r.metrics.aqi.kind).toBe("live");
  }
});
it("recorded official snapshots retain provenance and component margins, without credentials", () => {
  const raw = readFileSync(
      new URL("fixtures/acs-recorded.json", import.meta.url),
      "utf8",
    ),
    snapshot = JSON.parse(raw);
  expect(raw).not.toMatch(/[?&]key=/i);
  expect(snapshot.status).toBe("loaded");
  expect(Object.keys(snapshot.cities)).toHaveLength(5);
  for (const city of cities) {
    const m = snapshot.cities[city.id].metrics;
    expect(m.population.value).toBeGreaterThan(0);
    expect(m.rentBurden.components[7].marginOfError).not.toBeNull();
    expect(m.homeValue.table).toBe("B25077");
  }
});
