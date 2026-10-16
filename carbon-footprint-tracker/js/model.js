export const defaults = Object.freeze({ transport: "car", distance: 600, distanceUnit: "miles", period: "month", electricity: 650, gas: 25, flights: 0, waste: 8, diet: "average", household: 1, scope: "person" });
export const numericFields = ["distance", "electricity", "gas", "flights", "waste", "household"];
export const labels = { distance: "Commute distance", electricity: "Electricity", gas: "Natural gas", flights: "Flight distance", waste: "Landfill waste", household: "Household size" };
const choices = { transport: ["car","bus","train","bike","walk"], distanceUnit: ["miles","km"], period: ["week","month"], diet: ["meat","average","vegetarian","vegan"], scope: ["person","household"] };
export function validate(input) {
  const errors = {}, warnings = {};
  for (const key of numericFields) {
    const raw = input[key];
    if (raw === "" || raw === null || raw === undefined || typeof raw === "boolean" || !Number.isFinite(Number(raw))) errors[key] = "Enter a finite number; use 0 for none.";
    else if (Number(raw) < (key === "household" ? 1 : 0)) errors[key] = key === "household" ? "Enter at least 1 person." : "Enter 0 or greater.";
    else if (key === "household" && !Number.isInteger(Number(raw))) errors[key] = "Enter a whole number of people.";
    else if (Number(raw) > 10000000 || (key === "household" && Number(raw) > 1000)) errors[key] = "This exceeds the calculator's supported range.";
    else if (Number(raw) > ({distance:10000,electricity:10000,gas:500,flights:50000,waste:100,household:20}[key])) warnings[key] = "Unusually high — double-check the units. Included in your estimate.";
  }
  for (const [key, values] of Object.entries(choices)) if (!values.includes(input[key])) errors[key] = "Choose one of the listed options.";
  return {errors,warnings,valid:Object.keys(errors).length === 0};
}
export function calculate(input, data) {
  const check = validate(input);
  if (!check.valid) throw new RangeError("Invalid calculator inputs");
  const factors = Object.fromEntries(data.factors.map(f => [f.id, f]));
  const n = Number(input.household), personal = input.scope === "household" ? n : 1, shared = input.scope === "person" ? 1/n : 1;
  const miles = Number(input.distance) / (input.distanceUnit === "km" ? 1.609344 : 1) * (input.period === "week" ? 52/12 : 1);
  const flightMiles = Number(input.flights) / (input.distanceUnit === "km" ? 1.609344 : 1);
  const definitions = [
    ["transport","Transportation",input.transport,miles * personal],
    ["electricity","Electricity","electricity",Number(input.electricity)*shared],
    ["gas","Home heating","gas",Number(input.gas)*shared],
    ["flights","Air travel","flights",flightMiles*personal],
    ["waste","Waste","waste",Number(input.waste)*shared],
    ["diet","Diet",input.diet,30*personal]
  ];
  const categories = definitions.map(([id,label,factorId,quantity]) => {
    const factor = factors[factorId], value = quantity*factor.value, band = data.uncertainty[factor.confidence];
    return {id,label,factorId,quantity,value,low:value*(1-band),high:value*(1+band)};
  });
  const total = categories.reduce((a,c)=>a+c.value,0);
  return {categories,total,low:categories.reduce((a,c)=>a+c.low,0),high:categories.reduce((a,c)=>a+c.high,0),annual:total*12,tonnes:total/1000};
}
export function biggestLever(input, data) {
  const current = calculate(input,data);
  const candidates = [
    ["Switch your commute to walking or cycling", {...input,transport:"bike"}],
    ["Use 20% less electricity", {...input,electricity:Number(input.electricity)*.8}],
    ["Use 20% less natural gas", {...input,gas:Number(input.gas)*.8}],
    ["Skip this month's flights", {...input,flights:0}],
    ["Send 20% fewer bags to landfill", {...input,waste:Number(input.waste)*.8}],
    ["Choose a vegan diet", {...input,diet:"vegan"}]
  ].map(([label,next])=>({label,saving:current.total-calculate(next,data).total}));
  return candidates.sort((a,b)=>b.saving-a.saving)[0];
}

