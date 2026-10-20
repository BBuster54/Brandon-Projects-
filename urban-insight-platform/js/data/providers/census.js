import { numeric } from "../normalize.js";
export const VINTAGE = 2024;
export const TABLES = [
  "B01003",
  "B25077",
  "B25064",
  "B25070",
  "B08301",
  "B08303",
];
export function publicURL(value) {
  try {
    const u = new URL(value);
    for (const key of [...u.searchParams.keys()])
      if (/key|token|secret/i.test(key)) u.searchParams.delete(key);
    return u.href;
  } catch {
    return null;
  }
}
export function censusURL(city, table, key) {
  const url = new URL(`https://api.census.gov/data/${VINTAGE}/acs/acs5`);
  url.search = new URLSearchParams({
    get: `NAME,group(${table})`,
    for: `place:${city.fips.place}`,
    in: `state:${city.fips.state}`,
  });
  if (key) url.searchParams.set("key", key);
  return url;
}
export function parseTable(payload, city) {
  if (
    !Array.isArray(payload) ||
    payload.length !== 2 ||
    !Array.isArray(payload[0])
  )
    throw new Error("Invalid Census table");
  const row = Object.fromEntries(payload[0].map((k, i) => [k, payload[1][i]]));
  if (
    row.state !== city.fips.state ||
    row.place !== city.fips.place ||
    typeof row.NAME !== "string"
  )
    throw new Error("Census geography mismatch");
  return row;
}
export function normalizeCensus(city, tables, retrievedAt) {
  const base = (table) => ({
    kind: "official",
    vintage: VINTAGE,
    asOf: "2020–2024",
    retrievedAt,
    source: `US Census Bureau · ACS ${VINTAGE} 5-year · ${table}`,
    sourceUrl: publicURL(censusURL(city, table)),
    table,
    geography: { ...city.fips },
  });
  const cell = (table, bin) => {
    const row = tables[table] ?? {};
    const stem = `${table}_${String(bin).padStart(3, "0")}`;
    return {
      estimate: numeric(row[stem + "E"]),
      marginOfError: numeric(row[stem + "M"]),
      annotation: row[stem + "EA"] ?? null,
    };
  };
  const direct = (table, unit, method) => ({
    ...base(table),
    value: cell(table, 1).estimate,
    unit,
    marginOfError: cell(table, 1).marginOfError,
    method,
  });
  const share = (table, bins, excluded = [], method = "") => {
    const ids = [...new Set([1, ...bins, ...excluded])];
    const components = Object.fromEntries(
      ids.map((id) => [id, cell(table, id)]),
    );
    const valid = ids.every((id) => components[id].estimate !== null);
    const denominator = valid
      ? components[1].estimate -
        excluded.reduce((sum, id) => sum + components[id].estimate, 0)
      : 0;
    const numerator = valid
      ? bins.reduce((sum, id) => sum + components[id].estimate, 0)
      : 0;
    return {
      ...base(table),
      value:
        valid && denominator > 0 && numerator <= denominator
          ? (100 * numerator) / denominator
          : null,
      unit: "%",
      marginOfError: null,
      components,
      method:
        method +
        " Component published 90% margins are retained; no derived percentage margin is claimed.",
    };
  };
  const metrics = {
    population: direct(
      "B01003",
      "people",
      "Total resident population; five-year survey estimate.",
    ),
    homeValue: direct(
      "B25077",
      "USD",
      "Median homeowner-reported value of owner-occupied housing, not a market sale price.",
    ),
    rent: direct(
      "B25064",
      "USD/month",
      "Median monthly gross rent, including utilities.",
    ),
    rentBurden: share(
      "B25070",
      [7, 8, 9, 10],
      [11],
      "Sum of 30–34.9%, 35–39.9%, 40–49.9% and 50%+ bins, divided by total renter households minus not-computed households.",
    ),
  };
  for (const [key, bins] of Object.entries({
    car: [2],
    transit: [10],
    bike: [18],
    walk: [19],
    home: [21],
    other: [16, 17, 20],
  }))
    metrics[key] = share(
      "B08301",
      bins,
      [],
      "Share of workers age 16+ using this mode; working from home is included in the denominator.",
    );
  metrics.longCommute = share(
    "B08303",
    [12, 13],
    [],
    "Sum of 60–89 and 90+ minute bins divided by workers who commute. No mean travel time is inferred from bins.",
  );
  return { name: tables.B01003.NAME, metrics };
}
