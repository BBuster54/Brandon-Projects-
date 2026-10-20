import { node, format } from "./shared.js";
import { provenance } from "./provenance.js";
import { metricDefinitions, comparable } from "../data/normalize.js";
export function renderRank(cities, records, state) {
  const definition = metricDefinitions[state.metric];
  document.querySelector("#rank-title").textContent =
    definition.label + " across cities";
  const root = document.querySelector("#rank");
  root.replaceChildren();
  const rows = cities
    .map((c) => ({ city: c, metric: records[c.id]?.[state.metric] }))
    .filter((r) => comparable(r.metric))
    .sort(
      (a, b) =>
        (state.sort === "asc" ? 1 : -1) * (a.metric.value - b.metric.value),
    );
  if (!rows.length) {
    root.append(
      node(
        "p",
        state.metric === "aqi"
          ? "No current or last-good modeled readings for this filter."
          : "Official data not yet loaded for this metric.",
        "empty",
      ),
    );
    return;
  }
  const max = Math.max(1, ...rows.map((r) => r.metric.value));
  for (const { city, metric } of rows) {
    const row = node("div", undefined, "rank-row");
    row.append(
      node("span", city.name),
      node("strong", format(metric.value) + " " + metric.unit),
    );
    const track = node("span", undefined, "rank-track");
    track.setAttribute("aria-hidden", "true");
    const bar = node("span");
    bar.style.width = (metric.value / max) * 100 + "%";
    track.append(bar);
    row.append(track, provenance(metric, city));
    root.append(row);
  }
}
