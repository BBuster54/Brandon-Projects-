import { node, localTime } from "./shared.js";
import { publicURL } from "../data/providers/census.js";
export function provenance(metric, city) {
  const d = node("details", undefined, "provenance");
  const label =
    metric?.kind === "sample"
      ? "Source pending"
      : metric?.value === null
        ? metric?.kind === "official"
          ? "Not loaded"
          : "Unavailable"
        : metric?.kind === "live"
          ? metric.stale
            ? "Stale modeled"
            : "Live modeled"
          : metric?.value !== null && metric?.vintage
            ? "Official " + metric.vintage
            : "Not loaded";
  d.append(node("summary", label));
  const box = node("div");
  box.append(node("p", metric?.source ?? "No source available"));
  if (metric?.sourceUrl) {
    try {
      const u = new URL(publicURL(metric.sourceUrl));
      if (u.protocol === "https:") {
        const a = node("a", "Open source ↗");
        a.href = u.href;
        a.target = "_blank";
        a.rel = "noreferrer";
        box.append(a);
      }
    } catch {
      /* Do not render malformed URLs. */
    }
  }
  box.append(
    node(
      "p",
      metric?.kind === "official" && metric.vintage
        ? "ACS " +
            (metric.vintage - 4) +
            "–" +
            metric.vintage +
            " · Retrieved " +
            (metric.retrievedAt?.slice(0, 10) ?? "not yet")
        : localTime(metric?.asOf, city),
    ),
    node("p", metric?.method ?? ""),
  );
  if (metric?.marginOfError !== undefined && metric.marginOfError !== null)
    box.append(
      node(
        "p",
        "Published 90% margin of error: ±" +
          metric.marginOfError +
          " " +
          metric.unit,
      ),
    );
  d.append(box);
  return d;
}
export function renderMethods(cities, records, samples) {
  const section = document.querySelector("#method-content");
  section.replaceChildren();
  section.append(
    node(
      "p",
      "Air quality is modeled by CAMS global via Open-Meteo, on a coarse grid of about 45 km. A city-center grid cell can differ substantially from local monitors and does not represent every neighborhood.",
    ),
  );
  const airnow = node("a", "Use AirNow for official local monitor readings ↗");
  airnow.href = "https://www.airnow.gov/";
  airnow.target = "_blank";
  airnow.rel = "noreferrer";
  section.append(
    airnow,
    node(
      "p",
      "Census ACS figures are five-year survey estimates, with sampling uncertainty and different reference periods from current air quality. B25077 is a homeowner-reported estimate of home value, not a market sale price. Rent burden sums the 30–34.9%, 35–39.9%, 40–49.9% and 50%+ bins and divides by households with a computed rent-to-income ratio; 'not computed' households are excluded. Derived percentages are estimates, not exact counts; component margins of error are retained in the downloadable data.",
    ),
    node(
      "p",
      "Commute-mode shares use workers age 16+ as the denominator and include working from home. Long commutes use the travel-time table's commuter universe. Model, census and sample values are never combined into a synthetic score.",
    ),
  );
  const table = node("table"),
    caption = node("caption", "Metric provenance for the selected city");
  table.append(caption);
  const head = node("tr");
  for (const t of ["Metric", "Value / unit", "Source and freshness"])
    head.append(node("th", t));
  table.append(head);
  const c =
    cities.find((c) => c.id === document.querySelector("#city").value) ??
    cities[0];
  for (const [key, m] of Object.entries(records[c.id] ?? {})) {
    const row = node("tr");
    row.append(
      node("th", key),
      node(
        "td",
        m.value === null ? "Unavailable" : String(m.value) + " " + m.unit,
      ),
    );
    const td = node("td");
    td.append(provenance(m, c));
    row.append(td);
    table.append(row);
  }
  section.append(table);
  const illustrative = document.querySelector("#sample-content");
  illustrative.replaceChildren();
  illustrative.append(
    node(
      "p",
      "These old hand-entered values have no verified sources or measurement dates. They are excluded from headlines, comparisons and rankings. Sample AQI appears only as a labeled last-resort fallback, with a neutral marker and no current reading.",
    ),
  );
  for (const c of cities) {
    const p = node("p", c.name + ": ");
    for (const [key, m] of Object.entries(samples[c.id] ?? {}))
      if (key !== "aqi")
        p.append(
          node("span", key + " " + m.value + " " + m.unit + " "),
          node("span", "Source pending", "source-pending"),
          node("span", " · "),
        );
    illustrative.append(p);
  }
}
