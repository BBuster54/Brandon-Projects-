import { it, expect } from "vitest";
import { placeMapLabel } from "../js/render/map-label-layout.js";
it("places the New York label away from the Chicago marker", () => {
  const r = placeMapLabel({ x: 800, y: 260 }, [{ x: 640, y: 250 }], 1000, 650);
  expect(r.overlap).toBe(0);
  expect(r.y).toBeGreaterThan(280);
});
it("keeps edge annotations inside a phone map", () => {
  const r = placeMapLabel({ x: 300, y: 100 }, [{ x: 230, y: 100 }], 343, 360);
  expect(r.x).toBeGreaterThanOrEqual(8);
  expect(r.x + r.width).toBeLessThanOrEqual(335);
  expect(r.overlap).toBe(0);
});
it("avoids the docked filters as well as other markers", () => {
  const r = placeMapLabel({ x: 500, y: 170 }, [], 1000, 650, {
    blocked: [{ x: 16, y: 16, width: 968, height: 120 }],
  });
  expect(r.overlap).toBe(0);
});
