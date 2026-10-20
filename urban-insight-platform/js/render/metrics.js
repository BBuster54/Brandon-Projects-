import { node, format, localTime } from "./shared.js";
import { provenance } from "./provenance.js";
import { metricDefinitions, missingMetric } from "../data/normalize.js";
import { markerCategory } from "../aqi.js";
export function renderCity(city, record, view) {
  document.querySelector("#city-title").textContent = city.name;
  document.querySelector("#city-subtitle").textContent =
    city.state + " · " + city.timeZone;
  const aqi = record.aqi;
  const current = aqi?.kind === "live" && aqi.value !== null;
  document.querySelector("#aqi-value").textContent = current
    ? format(aqi.value, 0)
    : "—";
  document.querySelector("#aqi-status").textContent = current
    ? (markerCategory(aqi)?.label ?? "Unavailable") +
      (aqi.stale ? " · stale" : "")
    : "No current reading";
  document.querySelector("#aqi-time").textContent = current
    ? localTime(aqi.asOf, city)
    : "Modeled air quality not available";
  const source = document.querySelector("#headline-source");
  source.replaceChildren();
  if (aqi) source.append(provenance(aqi, city));
  const keys =
    view === "overview"
      ? ["population", "homeValue", "transit"]
      : Object.keys(metricDefinitions).filter(
          (k) => metricDefinitions[k].view === view && k !== "aqi",
        );
  const list = document.querySelector("#metric-list");
  list.replaceChildren();
  for (const key of keys) {
    const m = record[key] ?? missingMetric(key),
      d = metricDefinitions[key],
      row = node("article", undefined, "metric-row");
    row.append(
      node("h3", d.label),
      node(
        "p",
        m.value === null
          ? m.kind === "official"
            ? "Official data not yet loaded"
            : "Modeled data unavailable"
          : format(m.value) + (m.unit === "USD" ? " USD" : " " + m.unit),
        "metric-value",
      ),
      provenance(m, city),
    );
    list.append(row);
  }
}
export function renderCityTable(cities, records, select) {
  const focused = document.activeElement?.dataset.directoryCity;
  const body = document.querySelector("#city-table-body");
  body.replaceChildren();
  for (const c of cities) {
    const m = records[c.id]?.aqi,
      category = markerCategory(m),
      row = node("tr"),
      td = node("td"),
      button = node("button", c.name);
    button.dataset.directoryCity = c.id;
    button.addEventListener("click", () => select(c.id));
    td.append(button);
    row.append(
      td,
      node("td", m?.kind === "live" ? format(m.value, 0) : "—"),
      node(
        "td",
        category
          ? category.label + (m.stale ? " · stale" : "")
          : "No current reading",
      ),
    );
    const provenanceCell = node("td");
    provenanceCell.append(provenance(m, c));
    row.append(provenanceCell);
    body.append(row);
  }
  document.querySelector("#city-empty").hidden = cities.length > 0;
  if (focused)
    (
      body.querySelector(`[data-directory-city="${focused}"]`) ??
      document.querySelector("#city")
    ).focus({ preventScroll: true });
}
