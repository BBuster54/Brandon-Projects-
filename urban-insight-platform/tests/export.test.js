import { it, expect } from "vitest";
import { csv } from "../js/export.js";
it("exports only comparable values, provenance and no credentials; escapes formulas", () => {
  const output = csv([{ id: "a", name: "=TEST" }], {
    a: {
      aqi: {
        kind: "live",
        value: 20,
        unit: "US AQI",
        source: "CAMS",
        sourceUrl: "https://example.com/?key=private&get=value",
        asOf: "2026-10-20T12:00:00Z",
        stale: true,
      },
      sample: { kind: "sample", value: 999 },
    },
  });
  expect(output).not.toContain("private");
  expect(output).not.toContain("999");
  expect(output).toContain("'=TEST");
  expect(output).toContain('"true"');
});
