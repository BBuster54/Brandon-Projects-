import { describe, it, expect } from "vitest";
import { rampLayout } from "../js/render/ramp-layout.js";
describe("AQI annotation collision rules", () => {
  it("uses leader-line lanes when labels fit", () => {
    const r = rampLayout([0, 100, 200, 300, 500], 1100);
    expect(r.mode).toBe("staggered");
    expect(r.lanes).toBeLessThanOrEqual(3);
  });
  it("stacks narrow screens rather than compressing labels", () => {
    expect(rampLayout([51, 57, 64, 83, 95], 343).mode).toBe("stacked");
  });
  it("puts the known cluster into collision-free lanes", () => {
    const result = rampLayout([51, 57, 64, 83, 95], 1100);
    expect(result.mode).toBe("stacked");
    for (let i = 0; i < result.items.length; i++)
      for (let j = i + 1; j < result.items.length; j++) {
        const a = result.items[i],
          b = result.items[j];
        if (a.lane === b.lane)
          expect(Math.abs(a.labelX - b.labelX)).toBeGreaterThanOrEqual(212);
      }
  });
  it("keeps identical values at the same true position with distinct labels", () => {
    const result = rampLayout([64, 64, 64, 64, 64], 1100);
    expect(new Set(result.items.map((p) => p.x)).size).toBe(1);
    expect(new Set(result.items.map((p) => p.lane)).size).toBe(5);
  });
  it("falls back rather than losing labels if lanes are exhausted", () =>
    expect(rampLayout(Array(9).fill(60), 1100).mode).toBe("stacked"));
  it("keeps zero and 500 within the ramp and flags off-scale values", () => {
    const r = rampLayout([0, 50, 500, 600], 1100);
    expect(r.items[0].x).toBe(0);
    expect(r.items[2].x).toBe(1100);
    expect(r.items[3].offScale).toBe(true);
  });
});
