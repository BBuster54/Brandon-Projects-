import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
const base = process.env.TEST_URL || "http://localhost:4174/";
await mkdir("artifacts", { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text());
});
await page.goto(base);
await page.locator("#total").filter({ hasText: "862.1" }).waitFor();
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
    path: "artifacts/carbon-" + width + ".png",
    fullPage: true,
  });
}
await page.locator("#distance").fill("600");
await page.locator("#transport").selectOption("train");
await page.waitForFunction(
  () => document.querySelector("#total").textContent === "673.7",
);
await page.locator("#transport").selectOption("car");
await page.locator("#distanceUnit").selectOption("km");
assert.ok(
  Math.abs(Number(await page.locator("#distance").inputValue()) - 965.6064) <
    0.001,
);
await page.locator("#period").selectOption("week");
await page.waitForFunction(
  () => document.querySelector("#total").textContent === "862.1",
);
const commuteBeforeBlank = await page.locator("#distance").inputValue();
await page.locator("#distance").fill("");
await page.locator("#distanceUnit").selectOption("miles");
await page.locator("#distanceUnit").selectOption("km");
assert.equal(
  await page.locator("#distance").inputValue(),
  "",
  "Unit changes preserve blank input",
);
await page.locator("#distance").fill(commuteBeforeBlank);
await page.locator("#electricity").fill("");
await page.waitForTimeout(150);
assert.equal(
  await page.locator("#electricity").getAttribute("aria-invalid"),
  "true",
);
assert.equal(await page.locator("#export-csv").isDisabled(), true);
await page.locator("#electricity").fill("20000");
await page.waitForTimeout(150);
assert.match(
  await page.locator("#electricity-error").textContent(),
  /Unusually high/,
);
await page.locator("#electricity").fill("650");
await page.waitForTimeout(150);
await page.locator("#capture-baseline").click();
await page.locator('[data-preset="train"]').click();
assert.match(await page.locator("#scenario-delta").textContent(), /188.4/);
await page.locator("#household").fill("2");
await page.waitForTimeout(150);
assert.equal(await page.locator("#comparison-content").isVisible(), false);
await page.locator("#household").fill("1");
await page.waitForTimeout(150);
await page.locator("#goal").fill("700");
assert.equal(await page.locator("#goal-progress").isVisible(), true);
await page.locator("#save-month").click();
assert.equal(await page.locator("#history-list li").count(), 1);
const previousMonth = new Date(
  new Date().getFullYear(),
  new Date().getMonth() - 1,
  1,
);
await page
  .locator("#snapshot-month")
  .fill(
    previousMonth.getFullYear() +
      "-" +
      String(previousMonth.getMonth() + 1).padStart(2, "0"),
  );
await page.locator("#save-month").click();
assert.equal(await page.locator("#history-list li").count(), 2);
assert.equal(await page.locator("#trend").isVisible(), true);
await page.reload();
await page.waitForFunction(
  () => document.querySelector("#total").textContent !== "—",
);
assert.equal(await page.locator("#transport").inputValue(), "train");
assert.equal(await page.locator("#goal").inputValue(), "700");
for (const kind of ["csv", "json", "png"]) {
  const promise = page.waitForEvent("download");
  await page.locator("#export-" + kind).click();
  const d = await promise;
  await d.saveAs("artifacts/report." + kind);
}
await page.locator("#copy-link").click();
await page.waitForFunction(
  () => document.querySelector("#share-status").textContent.length > 0,
);
await page.locator("#clear-history").click();
await page.locator("#confirm-clear-no").click();
assert.equal(await page.locator("#history-list li").count(), 2);
await page.locator("#history-list button").first().click();
assert.equal(await page.locator("#history-list li").count(), 1);
await page.locator("#clear-history").click();
await page.locator("#confirm-clear-yes").click();
assert.equal(await page.locator("#history-list li").count(), 0);
await page.pdf({
  path: "artifacts/carbon-report.pdf",
  format: "A4",
  printBackground: true,
});
const blocked = await browser.newContext();
await blocked.addInitScript(() => {
  Object.defineProperty(window, "localStorage", {
    get() {
      throw new DOMException("blocked", "SecurityError");
    },
  });
});
const blockedPage = await blocked.newPage();
await blockedPage.goto(base);
await blockedPage.waitForFunction(
  () => document.querySelector("#total").textContent !== "—",
);
await blockedPage.locator("#save-month").click();
assert.match(
  await blockedPage.locator("#history-status").textContent(),
  /blocked/,
);
await blocked.close();
// A separate keyboard-only journey: no mouse clicks or programmatic focus.
await page.goto(base);
await page.waitForFunction(
  () => document.querySelector("#total").textContent !== "—",
);
const keyboardSteps = [];
await page.keyboard.press("Tab");
assert.equal(await page.locator(":focus").textContent(), "Skip to calculator");
await page.keyboard.press("Enter");
assert.equal(await page.locator(":focus").getAttribute("id"), "calculator");
keyboardSteps.push("Skip link moves focus to calculator.");
await page.keyboard.press("Tab");
assert.equal(await page.locator(":focus").getAttribute("id"), "household");
await page.keyboard.press(
  process.platform === "darwin" ? "Meta+A" : "Control+A",
);
await page.keyboard.type("2");
await page.keyboard.press("Enter");
await page.waitForTimeout(150);
assert.equal(await page.locator("#household").inputValue(), "2");
keyboardSteps.push(
  "Household input and Enter update results without navigation.",
);
let visited = [];
for (let i = 0; i < 65; i++) {
  await page.keyboard.press("Tab");
  const active = await page.evaluate(() => {
    const e = document.activeElement;
    return {
      id: e.id,
      text: e.tagName === "BODY" ? "Browser chrome" : e.textContent?.trim(),
      outline: getComputedStyle(e).outlineStyle,
    };
  });
  visited.push(active.id || active.text);
  if (active.text !== "Browser chrome")
    assert.notEqual(
      active.outline,
      "none",
      "Visible keyboard focus: " + JSON.stringify(active),
    );
  if (active.id === "transport") {
    await page.keyboard.press("b");
    await page.keyboard.press("Enter");
    assert.equal(await page.locator("#transport").inputValue(), "bus");
    await page.keyboard.press("c");
    await page.keyboard.press("Enter");
    keyboardSteps.push(
      "Native travel-mode select responds to keyboard selection.",
    );
  }
  if (active.id === "capture-baseline") {
    await page.keyboard.press("Enter");
    assert.equal(await page.locator("#comparison-content").isVisible(), true);
    keyboardSteps.push("Baseline button activates with Enter.");
  }
  if (active.id === "save-month") {
    await page.keyboard.press("Enter");
    keyboardSteps.push("Snapshot saves with Enter.");
  }
  if (active.id === "copy-link") {
    await page.keyboard.press("Enter");
    await page.waitForFunction(
      () => document.querySelector("#share-status").textContent.length > 0,
    );
    if (await page.locator("#share-fallback").isVisible()) {
      // Clipboard-denied flow deliberately focuses the selectable link.
      // Traverse back to Copy link, then continue forward through exports.
      for (let back = 0; back < 5; back++)
        await page.keyboard.press("Shift+Tab");
    }
    keyboardSteps.push("Copy link is keyboard accessible.");
  }
  if (active.id === "print-report")
    keyboardSteps.push("All export controls reached in tab order.");
  if (active.text === "Explore the model and emission factors") {
    await page.keyboard.press("Enter");
    assert.equal(await page.locator("details").getAttribute("open"), "");
    keyboardSteps.push("Methodology opens with Enter.");
    break;
  }
}
await writeFile(
  "artifacts/keyboard-order.json",
  JSON.stringify(visited, null, 2),
);
assert.ok(visited.includes("save-month"));
assert.ok(visited.includes("export-json"));
await page.emulateMedia({ colorScheme: "dark", reducedMotion: "reduce" });
await page.screenshot({ path: "artifacts/carbon-dark.png", fullPage: true });
assert.deepEqual(errors, [], "Browser console and runtime errors");
await writeFile(
  "artifacts/browser-results.json",
  JSON.stringify(
    {
      viewports: [375, 768, 1440],
      consoleErrors: errors,
      keyboardSteps,
      checks: [
        "known total",
        "unit conversion",
        "blank and high input",
        "scenario delta",
        "scope reset",
        "goals",
        "history and deletion",
        "URL restore",
        "CSV/JSON/PNG downloads",
        "clipboard feedback",
        "blocked storage",
        "print PDF",
      ],
    },
    null,
    2,
  ),
);
console.log(
  JSON.stringify({ passed: true, keyboardSteps, consoleErrors: errors }),
);
await browser.close();
