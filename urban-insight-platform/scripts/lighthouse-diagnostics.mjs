import { readdir, readFile } from "node:fs/promises";
let files;
try {
  files = (await readdir(".lighthouseci")).filter(
    (f) => f.startsWith("lhr-") && f.endsWith(".json"),
  );
} catch {
  console.log("No Lighthouse reports generated.");
  process.exit(0);
}
for (const file of files) {
  const d = JSON.parse(await readFile(".lighthouseci/" + file));
  console.log(
    JSON.stringify(
      {
        file,
        scores: Object.fromEntries(
          Object.entries(d.categories).map(([k, v]) => [k, v.score]),
        ),
        metrics: Object.fromEntries(
          [
            "first-contentful-paint",
            "largest-contentful-paint",
            "total-blocking-time",
            "cumulative-layout-shift",
          ].map((k) => [k, d.audits[k]?.numericValue]),
        ),
        mainThread: d.audits["mainthread-work-breakdown"]?.details?.items,
      },
      null,
      2,
    ),
  );
}
