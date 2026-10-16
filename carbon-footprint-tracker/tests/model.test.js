import { describe, it, expect } from "vitest";
import { defaults, calculate, validate, biggestLever } from "../js/model.js";
import factors from "../data/factors.json";
import fixtures from "./fixtures.json";
describe("shared known-answer fixtures",()=>{
  for(const f of fixtures) it(f.name,()=>{ const r=calculate(f.input,factors);expect(r.total).toBeCloseTo(f.total,8);expect(r.categories[0].value).toBeCloseTo(f.transport,8); });
  it("train swap saves 188.4 kg",()=>expect(calculate(fixtures[0].input,factors).total-calculate(fixtures[1].input,factors).total).toBeCloseTo(188.4));
  it("sums uncertainty bounds, without rounding categories",()=>{const r=calculate(defaults,factors);expect(r.low).toBeCloseTo(r.total*.5);expect(r.high).toBeCloseTo(r.total*1.5);});
  it("converts flight kilometers",()=>{const r=calculate({...defaults,flights:1609.344,distanceUnit:"km"},factors);expect(r.categories[3].value).toBeCloseTo(255);});
  it("names an actual modeled saving",()=>expect(biggestLever({...defaults,flights:10000},factors).saving).toBeCloseTo(2550));
});
describe("validation",()=>{
  for(const raw of ["",null,undefined,NaN,Infinity,-1,true]) it("rejects "+String(raw),()=>{expect(validate({...defaults,electricity:raw}).valid).toBe(false);expect(()=>calculate({...defaults,electricity:raw},factors)).toThrow();});
  it("warns about high usage without excluding it",()=>{const v=validate({...defaults,electricity:20000});expect(v.valid).toBe(true);expect(v.warnings.electricity).toBeTruthy();});
  it("rejects fractions and zero household size",()=>{expect(validate({...defaults,household:1.5}).valid).toBe(false);expect(validate({...defaults,household:0}).valid).toBe(false);});
  it("rejects unknown enums",()=>expect(validate({...defaults,transport:"spaceship"}).valid).toBe(false));
});

