import { describe, it, expect } from "vitest";
import { aqiCategory, markerCategory } from "../js/aqi.js";
import {
  numeric,
  selectFallback,
  filteredCities,
} from "../js/data/normalize.js";
import {
  parseState,
  stateQuery,
  CACHE_TTL,
  cacheFresh,
  storageRead,
  storageWrite,
} from "../js/state.js";
import { contrast } from "../js/contrast.js";
import citiesData from "../data/cities.json";
import palettes from "../data/theme-tokens.json";
describe("AQI categories", () => {
  for (const [v, label] of [
    [0, "good"],
    [50, "good"],
    [51, "moderate"],
    [100, "moderate"],
    [101, "sensitive"],
    [150, "sensitive"],
    [151, "unhealthy"],
    [200, "unhealthy"],
    [201, "very-unhealthy"],
    [300, "very-unhealthy"],
    [301, "hazardous"],
    [550, "hazardous"],
  ])
    it(String(v), () => expect(aqiCategory(v).id).toBe(label));
  for (const v of [null, undefined, -1, NaN, Infinity, "25"])
    it("missing " + v, () => expect(aqiCategory(v)).toBeNull());
  it("never uses EPA colors for sample markers", () =>
    expect(markerCategory({ value: 124, kind: "sample" })).toBeNull());
});
describe("fallback and normalization", () => {
  const now = Date.parse("2026-10-20T12:00:00Z"),
    live = { value: 52, kind: "live", asOf: "2026-10-20T12:00:00Z" },
    sample = { value: 99, kind: "sample", asOf: null };
  it("prefers fresh provider value", () =>
    expect(
      selectFallback(live, { ...live, value: 80 }, sample, now).value,
    ).toBe(52));
  it("uses last-good cache marked stale before samples", () => {
    const r = selectFallback(null, live, sample, now);
    expect(r.value).toBe(52);
    expect(r.stale).toBe(true);
  });
  it("falls back visibly to sample", () =>
    expect(selectFallback(null, null, sample, now).kind).toBe("sample"));
  it("does not turn missing data into zero", () => {
    for (const v of [null, "", undefined, NaN, -999999999])
      expect(numeric(v)).toBeNull();
    expect(numeric(0)).toBe(0);
  });
});
describe("state and cache", () => {
  it("TTL boundary and future clock", () => {
    const t = 1000000;
    expect(cacheFresh({ fetchedAt: t }, t + CACHE_TTL - 1)).toBe(true);
    expect(cacheFresh({ fetchedAt: t }, t + CACHE_TTL)).toBe(false);
    expect(cacheFresh({ fetchedAt: t }, t - 1)).toBe(false);
  });
  it("round trips and rejects unknown state", () => {
    const r = parseState(
      "?city=la&view=housing&metric=rent&sort=asc&compare=chi&category=good",
      citiesData.cities,
    );
    expect(parseState(stateQuery(r), citiesData.cities)).toEqual(r);
    expect(
      parseState("?view=<script>&city=invalid", citiesData.cities).city,
    ).toBe("nyc");
  });
  it("blocked storage is safe", () => {
    expect(storageRead(null, "x", "fallback")).toBe("fallback");
    expect(storageWrite(null, "x", {})).toBe(false);
  });
  it("excludes sample AQI from EPA category filters", () => {
    expect(
      filteredCities(
        citiesData.cities,
        { nyc: { aqi: { value: 20, kind: "sample" } } },
        "good",
      ),
    ).toHaveLength(0);
  });
});
describe("computed contrast", () => {
  for (const [mode, t] of Object.entries(palettes))
    for (const bg of ["paper", "surface"]) {
      for (const fg of ["ink", "muted", "accent"])
        it(mode + " " + fg + "/" + bg, () =>
          expect(contrast(t[fg], t[bg])).toBeGreaterThanOrEqual(4.5),
        );
      it(mode + " control/" + bg, () =>
        expect(contrast(t.control, t[bg])).toBeGreaterThanOrEqual(3),
      );
    }
});
