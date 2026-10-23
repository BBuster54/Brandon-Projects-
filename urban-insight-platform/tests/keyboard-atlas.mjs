import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { readFile, writeFile, mkdir } from "node:fs/promises";
const fixture = JSON.parse(
    await readFile(
      new URL("fixtures/open-meteo-recorded.json", import.meta.url),
    ),
  ),
  browser = await chromium.launch(),
  reports = [];
await mkdir("artifacts", { recursive: true });
for (const width of [375, 768, 1440]) {
  const page = await browser.newPage({ viewport: { width, height: 1000 } });
  await page.route("https://air-quality-api.open-meteo.com/**", (r) =>
    r.fulfill({ json: fixture }),
  );
  await page.route("https://tile.openstreetmap.org/**", (r) =>
    r.fulfill({
      contentType: "image/svg+xml",
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><path fill="#aaa" d="M0 0h256v256H0z"/></svg>',
    }),
  );
  await page.goto(process.env.TEST_URL || "http://localhost:4180/");
  await page.waitForFunction(
    () => !document.querySelector("#refresh").disabled,
  );
  const steps = [];
  async function reach(predicate, label) {
    for (let i = 0; i < 200; i++) {
      if (await page.evaluate(predicate)) {
        steps.push(label);
        return;
      }
      await page.keyboard.press("Tab");
    }
    throw new Error("Keyboard could not reach " + label);
  }
  await page.keyboard.press("Tab");
  assert.equal(
    await page.locator(":focus").textContent(),
    "Skip to city explorer",
  );
  await page.keyboard.press("Enter");
  assert.equal(await page.locator(":focus").getAttribute("id"), "explore");
  steps.push("Skip link");
  await reach(() => document.activeElement.id === "tab-overview", "View tabs");
  await page.keyboard.press("ArrowRight");
  assert.equal(
    await page.locator("#tab-environment").getAttribute("aria-selected"),
    "true",
  );
  await page.keyboard.press("End");
  assert.equal(
    await page.locator("#tab-housing").getAttribute("aria-selected"),
    "true",
  );
  await page.keyboard.press("Home");
  if (width < 600) {
    await reach(
      () => document.activeElement.parentElement?.id === "filter-disclosure",
      "Collapsed phone filter disclosure",
    );
    await page.keyboard.press("Enter");
  }
  await reach(() => document.activeElement.id === "city", "City selector");
  await page.keyboard.press("l");
  assert.equal(await page.locator("#city").inputValue(), "la");
  await reach(() => document.activeElement.id === "metric", "Metric selector");
  await page.keyboard.press("p");
  assert.equal(await page.locator("#metric").inputValue(), "pm25");
  await reach(
    () => document.activeElement.id === "category",
    "AQI category filter",
  );
  await page.keyboard.press("g");
  assert.match(await page.locator("#visible-count").textContent(), /0/);
  await reach(() => document.activeElement.id === "reset", "Reset");
  await page.keyboard.press("Enter");
  assert.equal(await page.locator("#category").inputValue(), "all");
  await reach(
    () => Boolean(document.activeElement.dataset.cityId),
    "Map marker",
  );
  const city = await page.locator(":focus").getAttribute("data-city-id");
  await page.keyboard.press("Space");
  assert.equal(await page.locator("#city").inputValue(), city);
  assert.equal(await page.locator(":focus").getAttribute("data-city-id"), city);
  await reach(
    () => Boolean(document.activeElement.dataset.directoryCity),
    "City directory",
  );
  await page.keyboard.press("Enter");
  assert.ok(await page.locator(":focus").getAttribute("data-directory-city"));
  await reach(
    () => document.activeElement.id === "compare",
    "Compare selector",
  );
  await page.keyboard.press("p");
  assert.equal(await page.locator("#compare").inputValue(), "phx");
  assert.match(page.url(), /compare=phx/);
  await reach(() => document.activeElement.id === "export", "CSV export");
  const download = page.waitForEvent("download");
  await page.keyboard.press("Enter");
  await (await download).saveAs(`artifacts/atlas-keyboard-${width}.csv`);
  await reach(() => document.activeElement.id === "share", "Share view");
  await page.keyboard.press("Enter");
  assert.ok((await page.locator("#export-status").textContent()).length > 0);
  await reach(
    () => document.activeElement.parentElement?.id === "methods-disclosure",
    "Methods disclosure",
  );
  await page.keyboard.press("Enter");
  assert.equal(
    await page.locator("#methods-disclosure").getAttribute("open"),
    "",
  );
  await page.keyboard.press("Enter");
  reports.push({
    width,
    result: "pass",
    input: "Keyboard events only; no clicks, focus calls, selectOption or fill",
    steps,
  });
  await page.close();
}
await writeFile(
  "artifacts/atlas-keyboard.json",
  JSON.stringify(reports, null, 2),
);
await browser.close();
console.log(
  "Keyboard-only operation passed at 375/768/1440: skip, tabs, disclosure, selectors, filtering/reset, marker and directory selection with retained focus, comparison/URL, export, share and methods.",
);
