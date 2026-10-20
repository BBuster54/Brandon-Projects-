import { readdir, readFile } from "node:fs/promises";
let files;
try {
  files = (await readdir(".lighthouseci")).filter(
    (name) => name.startsWith("lhr-") && name.endsWith(".json"),
  );
} catch {
  console.log("No Lighthouse reports were generated.");
  process.exit(0);
}
for (const file of files) {
  const report = JSON.parse(await readFile(".lighthouseci/" + file, "utf8"));
  console.log(
    JSON.stringify(
      {
        file,
        scores: Object.fromEntries(
          Object.entries(report.categories).map(([name, category]) => [
            name,
            category.score,
          ]),
        ),
        metrics: Object.fromEntries(
          [
            "first-contentful-paint",
            "largest-contentful-paint",
            "cumulative-layout-shift",
            "total-blocking-time",
            "speed-index",
          ].map((id) => [id, report.audits[id]?.numericValue]),
        ),
        layoutShifts: report.audits["layout-shifts"]?.details?.items,
        fontDisplay: report.audits["font-display"]?.details?.items,
        lcpElement: report.audits["largest-contentful-paint-element"]?.details,
        mainThread: report.audits["mainthread-work-breakdown"]?.details?.items,
        longTasks: report.audits["long-tasks"]?.details?.items,
      },
      null,
      2,
    ),
  );
}
