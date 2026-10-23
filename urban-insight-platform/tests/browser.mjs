import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { readFile, mkdir, writeFile } from "node:fs/promises";
const base = process.env.TEST_URL || "http://localhost:4180/";
const fixture = JSON.parse(
  await readFile(new URL("fixtures/open-meteo-recorded.json", import.meta.url)),
);
const browser = await chromium.launch();
await mkdir("artifacts", { recursive: true });
async function mock(context, { failure = false } = {}) {
  await context.route("https://air-quality-api.open-meteo.com/**", (route) =>
    failure
      ? route.fulfill({ status: 503, body: "Unavailable" })
      : route.fulfill({ json: fixture }),
  );
  await context.route("https://tile.openstreetmap.org/**", (route) =>
    route.fulfill({
      contentType: "image/svg+xml",
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><path fill="#e2e6e9" d="M0 0h256v256H0z"/><path d="M0 128h256M128 0v256" stroke="#c4cdd3"/></svg>',
    }),
  );
}
const context = await browser.newContext({
  viewport: { width: 1440, height: 1000 },
});
await mock(context);
const page = await context.newPage(),
  errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text());
});
await page.goto(base);
await page.waitForFunction(
  () => document.querySelector("#refresh").disabled === false,
);
assert.equal(await page.locator(".leaflet-marker-icon").count(), 5);
assert.notEqual(await page.locator("#aqi-value").textContent(), "—");
assert.match(await page.locator("#aqi-time").textContent(), /EDT|EST/);
assert.equal(
  await page.locator("#samples-disclosure").getAttribute("open"),
  null,
);
await page.locator("#refresh").click();
assert.match(await page.locator("#data-status").textContent(), /Cached/);
await page.locator("#city").selectOption("la");
assert.match(await page.locator("#aqi-time").textContent(), /PDT|PST/);
await page.locator("#compare").selectOption("chi");
assert.match(
  await page.locator("#comparison").textContent(),
  /Los Angeles.*Chicago/s,
);
await page.locator("#tab-overview").focus();
await page.keyboard.press("ArrowRight");
assert.equal(
  await page.locator("#tab-environment").getAttribute("aria-selected"),
  "true",
);
await page.locator("#tab-housing").click();
assert.match(
  await page.locator("#metric-list").textContent(),
  /owner-reported/,
);
await page.locator("#metric").selectOption("rent");
assert.match(await page.locator("#rank-title").textContent(), /rent/);
assert.ok((await page.locator("#rank .rank-row").count()) === 5);
assert.match(page.url(), /compare=chi/);
await page.reload();
await page.waitForFunction(
  () => document.querySelector("#refresh").disabled === false,
);
assert.equal(await page.locator("#city").inputValue(), "la");
assert.equal(await page.locator("#compare").inputValue(), "chi");
const downloadPromise = page.waitForEvent("download");
await page.locator("#export").click();
await (await downloadPromise).saveAs("artifacts/urban.csv");
await page.locator("#share").click();
assert.ok((await page.locator("#export-status").textContent()).length > 0);
await page.locator("#reset").click();
for (const width of [375, 768, 1440]) {
  await page.setViewportSize({ width, height: 1000 });
  await page.evaluate(() => document.fonts.ready);
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > innerWidth,
    ),
    false,
    "No horizontal overflow at " + width,
  );
  await page.screenshot({
    path: `artifacts/urban-fixture-${width}.png`,
    fullPage: true,
  });
}
await page.locator("#theme").click();
await page.screenshot({ path: "artifacts/urban-dark.png", fullPage: true });
// Keyboard-only traversal from page entry: no mouse actions in this pass.
await page.goto(base);
await page.waitForFunction(
  () => document.querySelector("#refresh").disabled === false,
);
await page.keyboard.press("Tab");
assert.equal(
  await page.locator(":focus").textContent(),
  "Skip to city explorer",
);
await page.keyboard.press("Enter");
assert.equal(await page.locator(":focus").getAttribute("id"), "explore");
const visited = [];
for (let i = 0; i < 160; i++) {
  await page.keyboard.press("Tab");
  const target = await page.evaluate(() => {
    const el = document.activeElement;
    return {
      id: el.id,
      role: el.getAttribute("role"),
      city: el.dataset.cityId,
      text: el.textContent.slice(0, 60),
    };
  });
  visited.push(target);
  if (target.city) {
    await page.keyboard.press("Space");
    assert.equal(
      await page.locator(":focus").getAttribute("data-city-id"),
      target.city,
    );
  }
  if (target.id === "samples-disclosure") await page.keyboard.press("Enter");
}
for (const id of [
  "city",
  "metric",
  "category",
  "reset",
  "refresh",
  "compare",
  "export",
  "share",
  "print",
])
  assert.ok(
    visited.some((t) => t.id === id),
    "Keyboard reaches " + id,
  );
await writeFile(
  "artifacts/keyboard.json",
  JSON.stringify({ result: "pass", visited }, null, 2),
);
// First-load API failure: no EPA-colored sample markers or invented headline values.
const failed = await browser.newContext();
await mock(failed, { failure: true });
const offline = await failed.newPage();
await offline.goto(base);
await offline.waitForFunction(
  () => document.querySelector("#refresh").disabled === false,
);
assert.equal(await offline.locator(".sample-marker").count(), 5);
assert.equal(await offline.locator("#aqi-value").textContent(), "—");
assert.match(
  await offline.locator("#data-status").textContent(),
  /Sample fallback/,
);
assert.equal(await offline.locator("#rank .rank-row").count(), 0);
await offline.screenshot({
  path: "artifacts/urban-api-failure.png",
  fullPage: true,
});
// Expired last-good cache survives an outage, retaining live kind and stale label.
await failed.addInitScript(
  ({ fixture }) => {
    const ids = ["nyc", "la", "chi", "hou", "phx"];
    const byCity = Object.fromEntries(
      ids.map((id, i) => [
        id,
        {
          metrics: {
            aqi: {
              value: fixture[i].current.us_aqi,
              unit: "US AQI",
              kind: "live",
              source: "Recorded CAMS model",
              asOf: new Date(fixture[i].current.time * 1000).toISOString(),
            },
          },
          trend: [],
        },
      ]),
    );
    sessionStorage.setItem(
      "urban-air-v2",
      JSON.stringify({ fetchedAt: Date.now() - 16 * 60000, byCity }),
    );
  },
  { fixture },
);
const stale = await failed.newPage();
await stale.goto(base);
await stale.waitForFunction(
  () => document.querySelector("#refresh").disabled === false,
);
assert.match(await stale.locator("#aqi-status").textContent(), /stale/);
assert.equal(await stale.locator(".sample-marker").count(), 0);
// Storage denial must not prevent controls from working.
const missing = await browser.newContext();
await mock(missing);
await missing.route("**/data/acs.json", (route) =>
  route.fulfill({ status: 503, body: "Unavailable" }),
);
const missingPage = await missing.newPage();
await missingPage.goto(base);
await missingPage.waitForFunction(
  () => document.querySelector("#refresh").disabled === false,
);
assert.match(
  await missingPage.locator("#metric-list").textContent(),
  /Official data not yet loaded/,
);
assert.notEqual(await missingPage.locator("#aqi-value").textContent(), "—");
const blocked = await browser.newContext();
await mock(blocked);
await blocked.addInitScript(() => {
  for (const key of ["localStorage", "sessionStorage"])
    Object.defineProperty(window, key, {
      get() {
        throw new DOMException("Blocked", "SecurityError");
      },
    });
});
const blockedPage = await blocked.newPage();
await blockedPage.goto(base);
await blockedPage.waitForFunction(
  () => document.querySelector("#refresh").disabled === false,
);
await blockedPage.locator("#city").selectOption("chi");
assert.equal(await blockedPage.locator("#city-title").textContent(), "Chicago");
const mobile = await browser.newContext({
  viewport: { width: 375, height: 800 },
  isMobile: true,
  hasTouch: true,
});
await mock(mobile);
const touch = await mobile.newPage();
await touch.goto(base);
await touch.waitForFunction(
  () => document.querySelector("#refresh").disabled === false,
);
await touch.locator("#map").scrollIntoViewIfNeeded();
const y = await touch.evaluate(() => scrollY);
const box = await touch.locator("#map").boundingBox();
const cdp = await mobile.newCDPSession(touch);
await cdp.send("Input.dispatchTouchEvent", {
  type: "touchStart",
  touchPoints: [{ x: box.x + 120, y: box.y + 220 }],
});
await cdp.send("Input.dispatchTouchEvent", {
  type: "touchMove",
  touchPoints: [{ x: box.x + 120, y: box.y + 70 }],
});
await cdp.send("Input.dispatchTouchEvent", {
  type: "touchEnd",
  touchPoints: [],
});
await touch.waitForTimeout(300);
assert.ok(
  (await touch.evaluate(() => scrollY)) > y,
  "One-finger map swipe scrolls page",
);
assert.deepEqual(
  errors,
  [],
  "No console or runtime errors on successful provider response",
);
console.log(
  "Playwright passed: three responsive widths, URL/compare/CSV, ARIA tabs, keyboard-only pass, API failure, stale cache, blocked storage and one-finger touch scroll.",
);
await browser.close();
