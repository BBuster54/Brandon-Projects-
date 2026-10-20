export const metricDefinitions = {
  aqi: { label: "US Air Quality Index", unit: "US AQI", view: "environment" },
  pm25: { label: "PM₂.₅", unit: "µg/m³", view: "environment" },
  pm10: { label: "PM₁₀", unit: "µg/m³", view: "environment" },
  ozone: { label: "Ozone", unit: "µg/m³", view: "environment" },
  no2: { label: "Nitrogen dioxide", unit: "µg/m³", view: "environment" },
  so2: { label: "Sulphur dioxide", unit: "µg/m³", view: "environment" },
  co: { label: "Carbon monoxide", unit: "µg/m³", view: "environment" },
  population: { label: "Population", unit: "people", view: "overview" },
  homeValue: {
    label: "Median owner-reported home value",
    unit: "USD",
    view: "housing",
  },
  rent: { label: "Median gross rent", unit: "USD/month", view: "housing" },
  rentBurden: {
    label: "Renters spending 30%+ on gross rent",
    unit: "%",
    view: "housing",
  },
  car: {
    label: "Car, truck or van commute",
    unit: "%",
    view: "transportation",
  },
  transit: {
    label: "Public transport commute",
    unit: "%",
    view: "transportation",
  },
  bike: { label: "Bicycle commute", unit: "%", view: "transportation" },
  walk: { label: "Walking commute", unit: "%", view: "transportation" },
  home: { label: "Work from home", unit: "%", view: "transportation" },
  other: { label: "Other commute modes", unit: "%", view: "transportation" },
  longCommute: {
    label: "Commutes of 60+ minutes",
    unit: "%",
    view: "transportation",
  },
};
export function numeric(value) {
  if (
    value === null ||
    value === undefined ||
    value === "" ||
    typeof value === "boolean"
  )
    return null;
  const n = Number(value);
  return Number.isFinite(n) && n >= 0 ? n : null;
}
export function missingMetric(key) {
  const air = metricDefinitions[key]?.view === "environment";
  return {
    value: null,
    unit: metricDefinitions[key]?.unit ?? "",
    source: air
      ? "Modeled air quality unavailable"
      : "Official data not yet loaded",
    sourceUrl: null,
    asOf: null,
    kind: air ? "live" : "official",
    vintage: null,
    method: air
      ? "No current or last-good model value is available."
      : "Run the Census ingestion script to load verified estimates.",
  };
}
export function comparable(metric) {
  return Boolean(
    metric &&
      metric.kind !== "sample" &&
      typeof metric.value === "number" &&
      Number.isFinite(metric.value),
  );
}
export function selectFallback(current, cached, sample, now = Date.now()) {
  const valid = (m) =>
    m?.kind === "live" &&
    numeric(m.value) !== null &&
    m.asOf &&
    Number.isFinite(Date.parse(m.asOf));
  if (valid(current))
    return { ...current, stale: now - Date.parse(current.asOf) > 90 * 60000 };
  if (valid(cached)) return { ...cached, stale: true };
  return sample ? { ...sample, stale: false } : null;
}
export function filteredCities(cities, records, category) {
  return category === "all"
    ? cities
    : cities.filter((c) => {
        const m = records[c.id]?.aqi;
        if (m?.kind !== "live" || m.value === null)
          return category === "unavailable";
        const v = m.value;
        const id =
          v <= 50
            ? "good"
            : v <= 100
              ? "moderate"
              : v <= 150
                ? "sensitive"
                : v <= 200
                  ? "unhealthy"
                  : v <= 300
                    ? "very-unhealthy"
                    : "hazardous";
        return id === category;
      });
}
