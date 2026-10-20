import { node, format, localTime } from "./shared.js";
import { metricDefinitions, comparable } from "../data/normalize.js";
import { provenance } from "./provenance.js";
export function renderCompare(cities, records, state) {
  const root = document.querySelector("#comparison"),
    a = cities.find((c) => c.id === state.city),
    b = cities.find((c) => c.id === state.compare);
  root.replaceChildren();
  document.querySelector("#compare").value = state.compare;
  if (a.id === b.id) {
    root.append(node("p", "Choose a different second city to compare."));
    return;
  }
  const table = node("table"),
    head = node("tr");
  for (const label of ["Metric", a.name, b.name]) {
    const th = node("th", label);
    th.scope = "col";
    head.append(th);
  }
  table.append(head);
  const keys = Object.keys(metricDefinitions).filter((k) =>
    state.view === "overview"
      ? ["aqi", "population", "homeValue", "rentBurden", "transit"].includes(k)
      : metricDefinitions[k].view === state.view,
  );
  for (const key of keys) {
    const row = node("tr"),
      th = node("th", metricDefinitions[key].label);
    th.scope = "row";
    row.append(th);
    for (const c of [a, b]) {
      const m = records[c.id][key],
        td = node("td");
      td.append(
        node(
          "span",
          comparable(m)
            ? format(m.value) + " " + m.unit
            : m.kind === "official"
              ? "Official data not yet loaded"
              : "No current reading",
        ),
        provenance(m, c),
      );
      row.append(td);
    }
    table.append(row);
  }
  root.append(table);
}
export function renderTrend(city, points, metric) {
  const root = document.querySelector("#trend");
  root.replaceChildren();
  document.querySelector("#trend-title").textContent =
    city.name + " · recent modeled AQI";
  if (!points.length) {
    root.append(
      node(
        "p",
        "No recent modeled history is available. Sample points are not plotted.",
      ),
    );
    return;
  }
  root.append(
    node(
      "p",
      "Up to 24 recent hourly model estimates, not monitor observations." +
        (metric.stale ? " Last-good history · stale." : ""),
    ),
    provenance(metric, city),
  );
  const ns = "http://www.w3.org/2000/svg",
    svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 800 160");
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    "Hourly modeled AQI trend. Exact values and city-local times are in the table below.",
  );
  const max = Math.max(50, ...points.map((p) => p.value)),
    polyline = document.createElementNS(ns, "polyline");
  polyline.setAttribute(
    "points",
    points
      .map(
        (p, i) =>
          `${12 + (i / Math.max(1, points.length - 1)) * 776},${145 - (p.value / max) * 130}`,
      )
      .join(" "),
  );
  polyline.setAttribute("fill", "none");
  polyline.setAttribute("stroke", "currentColor");
  polyline.setAttribute("stroke-width", "3");
  svg.append(polyline);
  root.append(svg);
  const details = node("details"),
    table = node("table"),
    head = node("tr");
  details.append(node("summary", "Hourly values · text alternative"));
  for (const label of ["City-local time and zone", "US AQI"]) {
    const th = node("th", label);
    th.scope = "col";
    head.append(th);
  }
  table.append(head);
  for (const p of points) {
    const row = node("tr");
    row.append(
      node("td", localTime(p.time, city)),
      node("td", format(p.value)),
    );
    table.append(row);
  }
  details.append(table);
  root.append(details);
}
