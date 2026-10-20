import { readFile, writeFile, rename } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve, dirname } from "node:path";
import {
  TABLES,
  VINTAGE,
  censusURL,
  parseTable,
  normalizeCensus,
} from "../js/data/providers/census.js";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
// Node 22 reads only conventional CENSUS_API_KEY from the local ignored .env.
try {
  process.loadEnvFile(resolve(root, ".env"));
} catch {
  /* Environment variables can be provided by the caller. */
}
const key = process.env.CENSUS_API_KEY;
if (!key) {
  console.log(
    "No Census key configured. Official data not yet loaded. Set CENSUS_API_KEY and rerun; no output was overwritten.",
  );
  process.exit(0);
}
try {
  const { cities } = JSON.parse(
    await readFile(resolve(root, "data/cities.json"), "utf8"),
  );
  const output = {
    version: "2.0.0",
    status: "loaded",
    vintage: VINTAGE,
    retrievedAt: new Date().toISOString(),
    tables: TABLES,
    cities: {},
  };
  // At most one request at a time; never log a private request URL or raw fetch error.
  for (const city of cities) {
    const tables = {};
    for (const table of TABLES) {
      const response = await fetch(censusURL(city, table, key), {
        signal: AbortSignal.timeout(15000),
      });
      if (!response.ok) throw new Error("Census request failed");
      tables[table] = parseTable(await response.json(), city);
    }
    output.cities[city.id] = normalizeCensus(city, tables, output.retrievedAt);
    console.log(`Validated official tables for ${city.name}.`);
  }
  const serialized = JSON.stringify(output, null, 2) + "\n";
  if (serialized.includes(key) || /[?&]key=/i.test(serialized))
    throw new Error("Credential guard");
  const target = resolve(root, "data/acs.json");
  await writeFile(target + ".tmp", serialized);
  await rename(target + ".tmp", target);
  console.log(
    "Saved five official city snapshots. Credentials are excluded from output.",
  );
} catch {
  console.error(
    "Census ingestion did not complete. Existing data was preserved. Check the configured key and network, then retry; no private request details are logged.",
  );
  process.exitCode = 1;
}
