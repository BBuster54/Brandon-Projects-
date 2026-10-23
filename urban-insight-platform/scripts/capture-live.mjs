import { chromium } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
const browser = await chromium.launch(),
  page = await browser.newPage({ viewport: { width: 1440, height: 1000 } }),
  errors = [];
page.on("pageerror", (e) => errors.push(e.message));
await mkdir("artifacts", { recursive: true });
await page.goto(process.env.TEST_URL || "http://localhost:4180/");
await page.waitForFunction(
  () => document.querySelector("#refresh").disabled === false,
);
await page.evaluate(() => document.fonts.ready);
for (const width of [375, 768, 1440]) {
  await page.setViewportSize({ width, height: 1000 });
  await page.waitForTimeout(800);
  await page.screenshot({
    path: `artifacts/urban-live-${width}.png`,
    fullPage: true,
  });
  await page.screenshot({ path: `artifacts/urban-live-${width}-viewport.png` });
}
await page.pdf({
  path: "artifacts/urban-print.pdf",
  format: "A4",
  printBackground: true,
});
await writeFile(
  "artifacts/live-status.json",
  JSON.stringify(
    {
      url: page.url(),
      status: await page.locator("#data-status").textContent(),
      errors,
    },
    null,
    2,
  ),
);
console.log(
  "Saved three viewport screenshots and print report; runtime errors: " +
    errors.length,
);
await browser.close();
