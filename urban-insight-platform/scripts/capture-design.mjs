import { chromium } from "@playwright/test";
import { mkdir, readFile } from "node:fs/promises";
const stage = process.argv[2] || "after";
if (!["before", "after"].includes(stage))
  throw new Error("Use before or after");
const fixture = JSON.parse(
  await readFile(
    new URL("../tests/fixtures/open-meteo-recorded.json", import.meta.url),
  ),
);
const browser = await chromium.launch();
await mkdir("docs/screenshots", { recursive: true });
for (const width of [375, 768, 1440]) {
  const page = await browser.newPage({ viewport: { width, height: 1000 } });
  await page.route("https://air-quality-api.open-meteo.com/**", (route) =>
    route.fulfill({ json: fixture }),
  );
  await page.goto(process.env.TEST_URL || "http://localhost:4180/");
  await page.waitForFunction(
    () => !document.querySelector("#refresh").disabled,
  );
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(1200);
  await page.screenshot({
    path: `docs/screenshots/atlas-${stage}-${width}.png`,
    fullPage: true,
  });
  await page.screenshot({
    path: `docs/screenshots/atlas-${stage}-${width}-viewport.png`,
  });
  if (width === 1440) {
    await page.evaluate(
      () => (document.documentElement.dataset.theme = "dark"),
    );
    await page.screenshot({
      path: `docs/screenshots/atlas-${stage}-dark.png`,
      fullPage: true,
    });
  }
  await page.close();
}
await browser.close();
console.log(
  `Captured ${stage} at 375/768/1440 with the same unmodified recorded air response and live OSM tiles.`,
);
