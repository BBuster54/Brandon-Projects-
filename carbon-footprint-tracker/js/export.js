import { calculate, validate } from "./model.js";
import { format, colors } from "./render.js";
export function reportPayload(
  input,
  data,
  baseline = null,
  goal = "",
  timestamp = new Date().toISOString(),
) {
  if (goal !== "" && (!Number.isFinite(Number(goal)) || Number(goal) <= 0 || Number(goal) > 10000000000)) {
    throw new RangeError("Enter a valid monthly target before exporting.");
  }
  return {
    timestamp,
    factorVersion: data.version,
    scope: input.scope,
    inputs: input,
    units: {
      distance: input.distanceUnit,
      commutePeriod: input.period,
      electricity: "kWh/month",
      gas: "therms/month",
      flights: input.distanceUnit + "/month",
      waste: "bags/month",
      result: "kg CO2e/month",
    },
    result: calculate(input, data),
    baseline: baseline
      ? { inputs: baseline, result: calculate(baseline, data) }
      : null,
    goal: goal === "" ? null : Number(goal),
    factors: data.factors,
    uncertainty: {
      method: "summed category bounds; assumed sensitivity, not statistical",
      bands: data.uncertainty,
    },
  };
}
const quote = (value) => '"' + String(value ?? "").replaceAll('"', '""') + '"';
export function reportCSV(payload) {
  const rows = [
    ["kind", "name", "value", "unit"],
    ["metadata", "timestamp", payload.timestamp, "ISO 8601"],
    ["metadata", "factor version", payload.factorVersion, ""],
  ];
  for (const [key, value] of Object.entries(payload.inputs))
    rows.push(["input", key, value, payload.units[key] || ""]);
  for (const c of payload.result.categories)
    rows.push(
      ["result", c.label, c.value, "kg CO2e/month"],
      ["lower bound", c.label, c.low, "kg CO2e/month"],
      ["upper bound", c.label, c.high, "kg CO2e/month"],
    );
  rows.push(
    ["result", "total", payload.result.total, "kg CO2e/month"],
    ["metadata", "uncertainty", payload.uncertainty.method, ""],
    ["goal", "monthly target", payload.goal, "kg CO2e/month"],
  );
  if (payload.baseline) {
    for (const [key, value] of Object.entries(payload.baseline.inputs))
      rows.push(["baseline input", key, value, ""]);
    for (const c of payload.baseline.result.categories)
      rows.push(["baseline result", c.label, c.value, "kg CO2e/month"]);
  }
  for (const f of payload.factors)
    rows.push(
      ["factor", f.id, f.value, f.unit],
      ["source", f.id, f.source, ""],
      ["confidence", f.id, f.confidence, ""],
    );
  return rows.map((r) => r.map(quote).join(",")).join("\r\n");
}
function download(blob, name) {
  const url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function initExports({ getInput, update, data, scenarios }) {
  let result = null;
  const status = document.querySelector("#export-status");
  function payload() {
    return reportPayload(
      getInput(),
      data,
      scenarios.baseline,
      document.querySelector("#goal").value,
    );
  }
  function validReport() {
    if (!validate(getInput()).valid) return false;
    const goal = document.querySelector("#goal");
    if (goal.getAttribute("aria-invalid") === "true") {
      status.textContent = "Fix or clear the monthly target before exporting.";
      goal.focus();
      return false;
    }
    return true;
  }
  for (const kind of ["csv", "json"])
    document.querySelector("#export-" + kind).addEventListener("click", () => {
      if (!validReport()) return;
      const report = payload(),
        contents =
          kind === "csv" ? reportCSV(report) : JSON.stringify(report, null, 2);
      download(
        new Blob([contents], {
          type: kind === "csv" ? "text/csv;charset=utf-8" : "application/json",
        }),
        "carbon-footprint." + kind,
      );
      status.textContent = kind.toUpperCase() + " report downloaded.";
    });
  document.querySelector("#export-png").addEventListener("click", () => {
    if (!result || !validReport()) return;
    result = calculate(getInput(), data);
    const canvas = document.createElement("canvas");
    canvas.width = 1400;
    canvas.height = 900;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      status.textContent = "Image export is unavailable in this browser.";
      return;
    }
    ctx.fillStyle = "#f3f6fc";
    ctx.fillRect(0, 0, 1400, 900);
    ctx.fillStyle = "#182d57";
    ctx.font = "600 42px Plex, Arial";
    ctx.fillText("Your monthly footprint", 70, 90);
    ctx.font = "600 70px Plex, Arial";
    ctx.fillText(format(result.total) + " kg CO₂e", 70, 200);
    ctx.font = "24px Plex, Arial";
    ctx.fillText(
      getInput().scope === "person" ? "Per person" : "Whole household",
      70,
      245,
    );
    const ordered = [...result.categories].sort((a, b) => b.value - a.value);
    let x = 70;
    for (const c of ordered) {
      const w = result.total ? (c.value / result.total) * 1260 : 0;
      ctx.fillStyle = colors[c.id];
      ctx.fillRect(x, 290, w, 100);
      x += w;
    }
    for (let i = 0; i < ordered.length; i++) {
      const c = ordered[i];
      ctx.fillStyle = colors[c.id];
      ctx.fillRect(70, 435 + i * 48, 18, 18);
      ctx.fillStyle = "#182d57";
      ctx.font = "24px Plex, Arial";
      ctx.fillText(c.label, 108, 453 + i * 48);
      ctx.fillText(
        format(c.value) +
          " kg (" +
          format(result.total ? (c.value / result.total) * 100 : 0) +
          "%)",
        750,
        453 + i * 48,
      );
    }
    ctx.font = "22px Plex, Arial";
    ctx.fillText(
      "Assumed range " +
        format(result.low) +
        "–" +
        format(result.high) +
        " kg · model " +
        data.version,
      70,
      790,
    );
    ctx.fillText(
      "Educational estimate · sources pending · " +
        new Date().toISOString().slice(0, 10),
      70,
      835,
    );
    canvas.toBlob((blob) => {
      if (blob) {
        download(blob, "carbon-footprint.png");
        status.textContent = "PNG chart downloaded.";
      } else status.textContent = "Image export failed. Try JSON or CSV.";
    }, "image/png");
  });
  document.querySelector("#print-report").addEventListener("click", () => {
    if (validate(getInput()).valid) {
      update();
      window.print();
    }
  });
  document.addEventListener("footprint-invalid", () => {
    for (const b of document.querySelectorAll("[data-export]"))
      b.disabled = true;
  });
  document.addEventListener("footprint-update", (e) => {
    result = e.detail.result;
    for (const b of document.querySelectorAll("[data-export]"))
      b.disabled = false;
    const i = e.detail.input;
    document.querySelector("#print-inputs").textContent =
      "Inputs: " +
      i.distance +
      " " +
      i.distanceUnit +
      "/" +
      i.period +
      " by " +
      i.transport +
      "; " +
      i.electricity +
      " kWh electricity; " +
      i.gas +
      " therms gas; " +
      i.flights +
      " " +
      i.distanceUnit +
      " flights; " +
      i.waste +
      " bags; " +
      i.diet +
      " diet; " +
      i.household +
      " people. Model " +
      data.version +
      ".";
  });
}
