import { it, expect } from "vitest";
import { defaults } from "../js/model.js";
import {
  encodeState,
  decodeState,
  sanitizeHistory,
  saveSnapshot,
  readStorage,
  writeStorage,
} from "../js/state.js";
import { reportPayload, reportCSV } from "../js/export.js";
import factors from "../data/factors.json";
it("round-trips every input, goal and baseline", () => {
  const result = decodeState(
    encodeState(defaults, "2.0.0", 700, { ...defaults, transport: "train" }),
    "2.0.0",
  );
  for (const key of Object.keys(defaults))
    expect(String(result.input[key])).toBe(String(defaults[key]));
  expect(result.goal).toBe("700");
  expect(result.baseline.transport).toBe("train");
});
it("invalid shared data falls back with feedback", () => {
  const r = decodeState("?distance=-1", "2.0.0");
  expect(r.input.distance).toBe(600);
  expect(r.message).toContain("invalid");
});
it("mismatched version warns", () =>
  expect(decodeState("?v=1", "2.0.0").message).toContain("different"));
it("corrupt history is discarded", () =>
  expect(sanitizeHistory([null, {}, { month: "2026-99" }])).toEqual([]));
it("storage failure is handled", () => {
  expect(readStorage("missing", [])).toEqual([]);
  expect(writeStorage("x", 1)).toBe(false);
});
it("same month/scope replaces, other scopes persist", () => {
  const a = { month: "2026-10", input: defaults, total: 800, version: "2" };
  const b = { ...a, total: 700 };
  expect(saveSnapshot([a], b)).toEqual([b]);
  expect(
    saveSnapshot([a], { ...b, input: { ...defaults, scope: "household" } }),
  ).toHaveLength(2);
});
it("report includes source factors, inputs, units and timestamp", () => {
  const p = reportPayload(
    defaults,
    factors,
    null,
    "700",
    "2026-10-16T00:00:00Z",
  );
  const csv = reportCSV(p);
  expect(p.result.total).toBeCloseTo(862.05);
  expect(csv).toContain('"factor","car","0.404"');
  expect(csv).toContain('"input","distance","600"');
  expect(csv).toContain('"factor version","2.0.0"');
  expect(csv).toContain("2026-10-16T00:00:00Z");
});
