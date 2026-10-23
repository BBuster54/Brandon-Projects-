import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { contrast } from "../js/contrast.js";
import { fillEdge } from "../js/render/epa-style.js";
import { categories } from "../js/aqi.js";
const fixture = JSON.parse(
  await readFile(new URL("fixtures/open-meteo-recorded.json", import.meta.url)),
);
const browser = await chromium.launch(),
  results = [];
async function hidden(page) {
  for (const id of ["why-built", "surprised"]) {
    assert.equal(await page.locator("#" + id).getAttribute("hidden"), "");
    assert.equal(
      await page.locator("#" + id).isVisible(),
      false,
      "Placeholder is visibly exposed: " + id,
    );
  }
}
const hex = (value) =>
  "#" +
  (value.match(/[\d.]+/g) || [])
    .slice(0, 3)
    .map((n) => Math.round(Number(n)).toString(16).padStart(2, "0"))
    .join("");
for (const width of [375, 768, 1440])
  for (const theme of ["light", "dark"]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } });
    await page.route("https://air-quality-api.open-meteo.com/**", (r) =>
      r.fulfill({ json: fixture }),
    );
    await page.route("https://tile.openstreetmap.org/**", (r) =>
      r.fulfill({
        contentType: "image/svg+xml",
        body: '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><path fill="#bfbfbf" d="M0 0h256v256H0z"/></svg>',
      }),
    );
    await page.goto(process.env.TEST_URL || "http://localhost:4180/");
    await page.waitForFunction(
      () => !document.querySelector("#refresh").disabled,
    );
    await page.evaluate(
      (theme) => (document.documentElement.dataset.theme = theme),
      theme,
    );
    await page.evaluate(() => document.fonts.ready);
    await hidden(page);
    if (width === 375 && theme === "light") {
      await page.evaluate(
        () => (document.querySelector("#why-built").hidden = false),
      );
      await assert.rejects(() => hidden(page), /Placeholder|Expected/);
      await page.evaluate(
        () => (document.querySelector("#why-built").hidden = true),
      );
    }
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth > innerWidth,
      ),
      false,
    );
    assert.equal(
      await page.locator("#rank").getAttribute("data-layout"),
      "stacked",
      "Clustered fixture must use the non-overlapping list fallback",
    );
    for (const field of [
      ".city-directory td[data-label=Source]",
      "#comparison td",
    ]) {
      const checks = await page.locator(field).evaluateAll((cells) =>
        cells.map((cell) => {
          const b = cell.getBoundingClientRect();
          return b.left >= 0 && b.right <= innerWidth;
        }),
      );
      assert.ok(
        checks.every(Boolean),
        "Mobile source/comparison cells fit viewport",
      );
    }
    assert.equal(
      await page.locator(".map-marker.selected .marker-name").isVisible(),
      true,
    );
    const other = page
      .locator(".city-marker-wrapper[aria-pressed=false]")
      .first();
    await other.focus();
    const ring = await other.evaluate(
      (el) => getComputedStyle(el).outlineColor,
    );
    assert.equal(hex(ring), theme === "light" ? "#000000" : "#ffffff");
    assert.equal(
      await page.locator(".marker-name:visible").count(),
      1,
      "Only one map label is visible during focus",
    );
    for (const selector of [
      ".leaflet-control-attribution",
      ".leaflet-marker-pane",
      ".leaflet-control-zoom",
    ])
      assert.equal(
        await page
          .locator(selector)
          .first()
          .evaluate((el) => getComputedStyle(el).filter),
        "none",
        "Only tile pane may be filtered",
      );
    const probes = await page.evaluate(() => {
      const canvas = document.createElement("canvas");
      canvas.width = 4;
      canvas.height = 2;
      const ctx = canvas.getContext("2d");
      ctx.filter = getComputedStyle(
        document.querySelector(".leaflet-tile-pane"),
      ).filter;
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, 2, 2);
      ctx.fillStyle = "#fff";
      ctx.fillRect(2, 0, 2, 2);
      return [
        Array.from(ctx.getImageData(0, 0, 1, 1).data).slice(0, 3),
        Array.from(ctx.getImageData(2, 0, 1, 1).data).slice(0, 3),
      ];
    });
    const edge = theme === "light" ? "#000000" : "#ffffff",
      ratios = probes.map((rgb) =>
        contrast(
          edge,
          "#" + rgb.map((x) => x.toString(16).padStart(2, "0")).join(""),
        ),
      );
    assert.ok(
      ratios.every((r) => r >= 3),
      "Marker boundary must contrast against both filtered tile extrema",
    );
    for (const category of categories)
      assert.ok(contrast(fillEdge(category.color), category.color) >= 3);
    const button = page.locator("#export"),
      states = {};
    for (const state of ["normal", "hover", "focus", "disabled"]) {
      await page.mouse.move(0, 0);
      await button.evaluate((el) => (el.disabled = false));
      if (state === "hover") await button.hover();
      if (state === "focus") {
        await page.keyboard.press("Tab");
        await button.focus();
      }
      if (state === "disabled")
        await button.evaluate((el) => (el.disabled = true));
      const colors = await button.evaluate((el) => {
        const s = getComputedStyle(el);
        return {
          fg: s.color,
          bg: s.backgroundColor,
          outline: s.outlineColor,
          visible: el.matches(":focus-visible"),
          height: el.getBoundingClientRect().height,
        };
      });
      const ratio = contrast(hex(colors.fg), hex(colors.bg));
      assert.ok(
        ratio >= (theme === "light" ? 5.7 : 8.57),
        `${state} text contrast ${theme}: ${ratio}`,
      );
      assert.ok(colors.height >= 44);
      if (state === "focus") assert.equal(colors.visible, true);
      states[state] = Number(ratio.toFixed(2));
    }
    if (width === 1440 && theme === "light") {
      const summary = page.locator("#trend > details > summary");
      await summary.click();
      await summary.focus();
      await page.setViewportSize({ width: 1430, height: 1000 });
      await page.waitForTimeout(150);
      assert.equal(
        await page.locator("#trend > details").getAttribute("open"),
        "",
      );
      assert.ok(
        await page
          .locator(":focus")
          .evaluate((el) => el.matches("#trend > details > summary")),
      );
      const rankSource = page.locator("#rank .provenance summary").first();
      await rankSource.click();
      await rankSource.focus();
      await page.setViewportSize({ width: 1440, height: 1000 });
      await page.waitForTimeout(150);
      assert.ok(
        await page
          .locator(":focus")
          .evaluate((el) => Boolean(el.closest("#rank .provenance[open]"))),
      );
    }
    if (width === 1440 && theme === "dark") {
      await page.evaluate(() => delete document.documentElement.dataset.theme);
      await page.emulateMedia({ media: "print", colorScheme: "dark" });
      const print = await page.locator("body").evaluate((el) => {
        const s = getComputedStyle(el);
        return { fg: s.color, bg: s.backgroundColor };
      });
      assert.ok(
        contrast(hex(print.fg), hex(print.bg)) >= 5.7,
        "System-dark print must remain dark ink on paper",
      );
      await hidden(page);
    }
    results.push({
      width,
      theme,
      tileFilterExtrema: probes,
      tileBoundaryContrast: ratios.map((r) => Number(r.toFixed(2))),
      filledButtonTextContrast: states,
      placeholderHidden: true,
      stackedTables: true,
    });
    await page.close();
  }
await mkdir("artifacts", { recursive: true });
await writeFile(
  "artifacts/atlas-presentation.json",
  JSON.stringify(results, null, 2),
);
await browser.close();
console.log(
  "Atlas presentation checks passed in both themes at 375/768/1440: placeholders (including a negative visibility test), table fit, label collisions, filtered-tile contrast, and filled button states.",
);
