import { node, format, localTime, measurement } from "./shared.js";
import { metricDefinitions, comparable } from "../data/normalize.js";
import { provenance } from "./provenance.js";
export function renderCompare(cities, records, state) {
  const root = document.querySelector("#comparison"),
    a = cities.find((c) => c.id === state.city),
    b = cities.find((c) => c.id === state.compare);
  root.replaceChildren();
  document.querySelector("#compare").value = state.compare;
  document.querySelector("#compare-source").textContent =
    state.view === "environment"
      ? "Modeled · CAMS / Open-Meteo"
      : state.view === "overview"
        ? "Modeled + official estimates"
        : "Official · Census ACS";
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
      td.dataset.label = c.name;
      const value = node("span");
      value.append(
        comparable(m)
          ? measurement(m.value, m.unit)
          : node(
              "span",
              m.kind === "official"
                ? "Official data not yet loaded"
                : "No current reading",
            ),
      );
      td.append(value, provenance(m, c, { compact: true }));
      row.append(td);
    }
    table.append(row);
  }
  root.append(table);
}
let lastTrend;
export function renderTrend(city, points, metric) {
  lastTrend = [city, points, metric];
  const root = document.querySelector("#trend");
  root.replaceChildren();
  document.querySelector("#trend-title").textContent =
    city.name + " · recent modeled AQI";
  const source = document.querySelector("#trend-source");
  source.replaceChildren();
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
  );
  source.append(provenance(metric, city));
  const ns = "http://www.w3.org/2000/svg",
    svg = document.createElementNS(ns, "svg");
  const width = Math.max(280, root.clientWidth),
    height = 260,
    left = 48,
    right = 16,
    top = 32,
    bottom = 38;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    "Hourly modeled AQI trend. Exact values and city-local times are in the table below.",
  );
  const max = Math.ceil(Math.max(50, ...points.map((p) => p.value)) / 25) * 25,
    polyline = document.createElementNS(ns, "polyline");
  const y = (value) =>
    height - bottom - (value / max) * (height - top - bottom);
  function label(text, x, y, className) {
    const t = document.createElementNS(ns, "text");
    t.textContent = text;
    t.setAttribute("x", x);
    t.setAttribute("y", y);
    t.setAttribute("class", className);
    svg.append(t);
    return t;
  }
  label("US AQI", 0, 17, "trend-axis-title");
  for (let n = 0; n <= 4; n++) {
    const value = (max * n) / 4,
      line = document.createElementNS(ns, "line");
    line.setAttribute("x1", left);
    line.setAttribute("x2", width - right);
    line.setAttribute("y1", y(value));
    line.setAttribute("y2", y(value));
    line.setAttribute("class", "trend-grid");
    svg.append(line);
    const text = label(
      format(value),
      left - 8,
      y(value) + 4,
      "figure trend-tick",
    );
    text.setAttribute("text-anchor", "end");
  }
  polyline.setAttribute(
    "points",
    points
      .map(
        (p, i) =>
          `${left + (i / Math.max(1, points.length - 1)) * (width - left - right)},${y(p.value)}`,
      )
      .join(" "),
  );
  polyline.setAttribute("fill", "none");
  polyline.setAttribute("stroke", "currentColor");
  polyline.setAttribute("stroke-width", "3");
  svg.append(polyline);
  root.append(svg);
  const range = node("div", undefined, "trend-range"),
    min = Math.min(...points.map((p) => p.value)),
    actualMax = Math.max(...points.map((p) => p.value));
  range.append(
    node("span", "Minimum "),
    measurement(min, "US AQI"),
    node("span", " · Maximum "),
    measurement(actualMax, "US AQI"),
  );
  root.append(range);
  const times = node("div", undefined, "trend-times");
  times.append(
    node("span", "Start · " + localTime(points[0].time, city), "timestamp"),
    node("span", "End · " + localTime(points.at(-1).time, city), "timestamp"),
  );
  root.append(times);
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
      node("td", localTime(p.time, city), "timestamp"),
      node("td", format(p.value), "figure"),
    );
    table.append(row);
  }
  details.append(table);
  root.append(details);
}
globalThis.addEventListener?.("resize", () => {
  if (!lastTrend) return;
  const tableOpen = document.querySelector("#trend > details")?.open,
    sourceOpen = document.querySelector("#trend-source details")?.open,
    active = document.activeElement,
    tableFocus = active.matches?.("#trend > details > summary"),
    sourceFocus = active.closest?.("#trend-source"),
    link = active.tagName === "A";
  renderTrend(...lastTrend);
  if (tableOpen)
    document.querySelector("#trend > details")?.setAttribute("open", "");
  if (sourceOpen)
    document.querySelector("#trend-source details")?.setAttribute("open", "");
  if (tableFocus)
    document
      .querySelector("#trend > details > summary")
      ?.focus({ preventScroll: true });
  if (sourceFocus)
    document
      .querySelector(`#trend-source ${link ? "a" : "summary"}`)
      ?.focus({ preventScroll: true });
});
