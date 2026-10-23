import { describe, it, expect } from "vitest";
import {
  censusURL,
  publicURL,
  parseTable,
  normalizeCensus,
} from "../js/data/providers/census.js";
const city = { fips: { state: "36", place: "51000" } };
// Synthetic arithmetic fixture: not a published city estimate and never shipped to the UI.
const rows = {};
for (const table of [
  "B01003",
  "B25077",
  "B25064",
  "B25070",
  "B08301",
  "B08303",
]) {
  rows[table] = { NAME: "Fixture city" };
  for (let n = 1; n <= 21; n++) {
    rows[table][`${table}_${String(n).padStart(3, "0")}E`] =
      n === 1 ? "100" : "5";
    rows[table][`${table}_${String(n).padStart(3, "0")}M`] = "2";
  }
}
describe("official ingestion", () => {
  it("strips private credentials before serialization", () => {
    const u = censusURL(city, "B25077", "unit-test-private");
    expect(u.href).toContain("key=");
    expect(publicURL(u)).not.toContain("private");
    expect(publicURL("https://api.census.gov/?KEY=x&token=x&get=NAME")).toBe(
      "https://api.census.gov/?get=NAME",
    );
    expect(
      JSON.stringify(normalizeCensus(city, rows, "2026-10-20T00:00:00Z")),
    ).not.toContain("key=");
  });
  it("validates FIPS", () => {
    expect(() =>
      parseTable(
        [
          ["NAME", "state", "place"],
          ["Fixture", "00", "51000"],
        ],
        city,
      ),
    ).toThrow();
    expect(
      parseTable(
        [
          ["NAME", "state", "place"],
          ["Fixture", "36", "51000"],
        ],
        city,
      ).NAME,
    ).toBe("Fixture");
  });
  it("sums rent bins and excludes not-computed households", () => {
    const m = normalizeCensus(city, rows, "now").metrics;
    expect(m.rentBurden.value).toBeCloseTo((20 / 95) * 100);
    expect(m.rentBurden.marginOfError).toBeNull();
    expect(m.rentBurden.components[7].marginOfError).toBe(2);
    expect(m.homeValue.method).toContain("not a market");
    expect(m.other.value).toBe(15);
    expect(m.longCommute.value).toBe(10);
  });
  it("treats sentinels as missing rather than zero", () => {
    const copy = structuredClone(rows);
    copy.B25077.B25077_001E = "-666666666";
    copy.B25070.B25070_007E = null;
    const m = normalizeCensus(city, copy, "now").metrics;
    expect(m.homeValue.value).toBeNull();
    expect(m.rentBurden.value).toBeNull();
  });
});
