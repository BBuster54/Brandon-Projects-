import { describe, it, expect } from "vitest";
import { categories } from "../js/aqi.js";
import { contrast } from "../js/contrast.js";
import { fillEdge, tileRange } from "../js/render/epa-style.js";
import tokens from "../data/theme-tokens.json";
describe("Atlas boundary and interaction contrast", () => {
  for (const [mode, t] of Object.entries(tokens)) {
    for (const c of categories)
      it(`${mode} ${c.label} fill keyline and outer page/tile boundary`, () => {
        expect(contrast(fillEdge(c.color), c.color)).toBeGreaterThanOrEqual(3);
        for (const bg of [
          t.paper,
          t.surface,
          ...Object.values(tileRange[mode]),
        ])
          expect(contrast(t.edge, bg)).toBeGreaterThanOrEqual(3);
      });
    for (const state of ["normal", "hover", "disabled", "focus"])
      it(`${mode} filled ${state} remains readable`, () =>
        expect(
          contrast(t.buttonText, state === "hover" ? t.ink : t.accent),
        ).toBeGreaterThanOrEqual(mode === "light" ? 5.7 : 8.57));
  }
});
