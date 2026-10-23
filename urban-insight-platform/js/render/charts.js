import { node, measurement } from "./shared.js";
import { provenance } from "./provenance.js";
import { metricDefinitions, comparable } from "../data/normalize.js";
import { categories, aqiCategory } from "../aqi.js";
import { applyEPA } from "./epa-style.js";
import { rampLayout } from "./ramp-layout.js";
let lastArgs;
function figure(value, unit) {
  const result = node("strong");
  result.append(measurement(value, unit));
  return result;
}
export function renderRank(cities, records, state) {
  lastArgs = [cities, records, state];
  const definition = metricDefinitions[state.metric],
    root = document.querySelector("#rank");
  root.replaceChildren();
  document.querySelector("#rank-title").textContent =
    state.metric === "aqi"
      ? "Air quality across five cities"
      : definition.label + " across cities";
  const rows = cities
    .map((city) => ({ city, metric: records[city.id]?.[state.metric] }))
    .filter((r) => comparable(r.metric))
    .sort(
      (a, b) =>
        (state.sort === "asc" ? 1 : -1) * (a.metric.value - b.metric.value),
    );
  document.querySelector("#rank-source").textContent =
    definition.view === "environment"
      ? "Modeled · CAMS / Open-Meteo"
      : rows[0]?.metric.vintage
        ? "Official · ACS " + rows[0].metric.vintage
        : "Official estimates";
  if (!rows.length) {
    root.append(
      node(
        "p",
        definition.view === "environment"
          ? "No current or last-good modeled readings for this filter."
          : "Official data not yet loaded for this metric.",
        "empty",
      ),
    );
    return;
  }
  if (state.metric !== "aqi") {
    const max = Math.max(1, ...rows.map((r) => r.metric.value));
    for (const { city, metric } of rows) {
      const row = node("div", undefined, "rank-row"),
        track = node("span", undefined, "rank-track"),
        bar = node("span");
      row.dataset.rankCity = city.id;
      row.append(
        node("span", city.name),
        figure(metric.value, metric.unit),
        provenance(metric, city, { compact: true }),
      );
      track.setAttribute("aria-hidden", "true");
      bar.style.width = (metric.value / max) * 100 + "%";
      track.append(bar);
      row.append(track);
      root.append(row);
    }
    return;
  }
  const width = root.clientWidth,
    layout = rampLayout(
      rows.map((r) => r.metric.value),
      width,
    );
  root.dataset.layout = layout.mode;
  const track = node("div", undefined, "epa-ramp");
  track.setAttribute("aria-hidden", "true");
  let start = 0;
  for (const c of categories) {
    const end = Math.min(500, c.max),
      segment = node("span", undefined, "epa-segment");
    segment.style.width = (end - start) / 5 + "%";
    applyEPA(segment, c.color);
    track.append(segment);
    start = end;
  }
  for (const p of layout.items) {
    const mark = node("i", undefined, "ramp-pin");
    mark.style.left = p.x + "px";
    applyEPA(mark, aqiCategory(p.value).color);
    track.append(mark);
  }
  root.append(track);
  const axis = node("div", undefined, "ramp-axis");
  axis.setAttribute("aria-hidden", "true");
  for (const value of [0, 50, 100, 150, 200, 300, 500]) {
    const tick = node("span", String(value), "figure");
    tick.style.left = value / 5 + "%";
    axis.append(tick);
  }
  root.append(axis);
  const legend = node("div", undefined, "ramp-key");
  for (const c of categories) {
    const key = node("span"),
      swatch = node("i");
    applyEPA(swatch, c.color);
    key.append(swatch, node("span", c.label));
    legend.append(key);
  }
  const list = node("div", undefined, "ramp-labels " + layout.mode);
  list.setAttribute("role", "list");
  if (layout.mode === "staggered") {
    list.style.height = layout.lanes * 90 + 30 + "px";
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${layout.lanes * 90 + 30}`);
    svg.setAttribute("aria-hidden", "true");
    svg.classList.add("ramp-leaders");
    for (const p of layout.items) {
      const line = document.createElementNS(svg.namespaceURI, "polyline");
      line.setAttribute(
        "points",
        `${p.x},-30 ${p.labelX},${p.lane * 90 + 14} ${p.labelX},${p.lane * 90 + 28}`,
      );
      svg.append(line);
    }
    list.append(svg);
  }
  rows.forEach(({ city, metric }, i) => {
    const p = layout.items[i],
      row = node("div", undefined, "rank-row ramp-city"),
      category = aqiCategory(metric.value);
    row.setAttribute("role", "listitem");
    row.dataset.rankCity = city.id;
    row.append(
      node("span", city.name),
      figure(metric.value, "US AQI"),
      node(
        "span",
        category.label +
          (metric.stale ? " · stale" : "") +
          (p.offScale ? " · above ramp range" : ""),
        "category-label",
      ),
      provenance(metric, city, { compact: true }),
    );
    if (layout.mode === "staggered") {
      row.classList.add("ramp-callout");
      row.style.left = p.labelX + "px";
      row.style.top = p.lane * 90 + 28 + "px";
    }
    list.append(row);
  });
  root.append(
    list,
    legend,
    node(
      "p",
      layout.mode === "stacked"
        ? "City readings are listed below to keep nearby values legible. Positions use a linear 0–500 US AQI scale."
        : "Leader lines connect each city to its exact position on the 0–500 US AQI scale.",
      "small",
    ),
  );
}
globalThis.addEventListener?.("resize", () => {
  if (!lastArgs) return;
  const root = document.querySelector("#rank"),
    active = document.activeElement,
    city = active.closest?.("[data-rank-city]")?.dataset.rankCity,
    link = active.tagName === "A";
  const open = [...root.querySelectorAll(".provenance[open]")].map(
    (d) => d.closest("[data-rank-city]")?.dataset.rankCity,
  );
  renderRank(...lastArgs);
  for (const id of open)
    root
      .querySelector(`[data-rank-city="${id}"] .provenance`)
      ?.setAttribute("open", "");
  if (city)
    root
      .querySelector(
        `[data-rank-city="${city}"] .provenance ${link ? "a" : "summary"}`,
      )
      ?.focus({ preventScroll: true });
});
