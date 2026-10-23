import { comparable, metricDefinitions } from "./data/normalize.js";
import { publicURL } from "./data/providers/census.js";
export function csv(cities, records, view = null) {
  const quote = (value) =>
    '"' +
    String(value ?? "")
      .replace(/^[=+@-]/, "'$&")
      .replaceAll('"', '""') +
    '"';
  const rows = [
    [
      "City",
      "Metric",
      "Value",
      "Unit",
      "Kind",
      "Stale",
      "Source",
      "Source URL",
      "As of",
      "Vintage",
      "Method",
    ],
  ];
  for (const c of cities)
    for (const [key, m] of Object.entries(records[c.id] ?? {}))
      if (
        comparable(m) &&
        (!view ||
          (view === "overview"
            ? [
                "aqi",
                "population",
                "homeValue",
                "rentBurden",
                "transit",
              ].includes(key)
            : metricDefinitions[key]?.view === view))
      )
        rows.push([
          c.name,
          metricDefinitions[key]?.label ?? key,
          m.value,
          m.unit,
          m.kind,
          Boolean(m.stale),
          m.source,
          publicURL(m.sourceUrl),
          m.asOf,
          m.vintage,
          m.method,
        ]);
  return rows.map((row) => row.map(quote).join(",")).join("\r\n");
}
export function downloadCSV(cities, records, view) {
  const blob = new Blob(["\ufeff" + csv(cities, records, view)], {
      type: "text/csv;charset=utf-8",
    }),
    url = URL.createObjectURL(blob),
    link = document.createElement("a");
  link.href = url;
  link.download = "urban-insight.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
