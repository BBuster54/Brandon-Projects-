import { firefox, webkit } from "@playwright/test";
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
const results = [];
for (const [name, engine] of [
  ["Firefox", firefox],
  ["WebKit", webkit],
]) {
  const browser = await engine.launch();
  const page = await browser.newPage({ viewport: { width: 375, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(process.env.TEST_URL || "http://localhost:4174/");
  await page.waitForFunction(
    () => document.querySelector("#total").textContent === "862.1",
  );
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
    false,
  );
  await page.locator('[data-preset="train"]').click();
  assert.match(await page.locator("#scenario-delta").textContent(), /188.4/);
  await page.locator("#save-month").click();
  assert.equal(await page.locator("#history-list li").count(), 1);
  await page.reload();
  await page.waitForFunction(
    () => document.querySelector("#total").textContent === "673.7",
  );
  assert.equal(await page.locator("#history-list li").count(), 1);
  await page.locator("#electricity").fill("");
  await page.waitForTimeout(200);
  assert.equal(
    await page.locator("#electricity").getAttribute("aria-invalid"),
    "true",
  );
  assert.deepEqual(errors, []);
  results.push({ name, version: browser.version(), passed: true });
  await browser.close();
}
await writeFile(
  "artifacts/cross-browser.json",
  JSON.stringify(results, null, 2),
);
console.log(JSON.stringify(results));
