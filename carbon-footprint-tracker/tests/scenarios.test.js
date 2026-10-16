import {it,expect} from "vitest";
import {defaults,calculate} from "../js/model.js";
import {presets,comparison} from "../js/scenarios.js";
import factors from "../data/factors.json";
it("train preset keeps original inputs unchanged and saves 188.4",()=>{const before={...defaults};const after=presets.train(before);expect(before.transport).toBe("car");expect(comparison(calculate(before,factors),calculate(after,factors))[0].saving).toBeCloseTo(188.4);});
it("electricity cut is exactly 20 percent",()=>{const rows=comparison(calculate(defaults,factors),calculate(presets.electricity(defaults),factors));expect(rows[1].percent).toBeCloseTo(20);});
it("zero baseline has no percentage",()=>{const rows=comparison(calculate({...defaults,electricity:0},factors),calculate(defaults,factors));expect(rows[1].percent).toBeNull();expect(rows[1].saving).toBeLessThan(0);});

