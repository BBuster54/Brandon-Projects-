import { readFile, readdir, stat } from "node:fs/promises";
import assert from "node:assert/strict";
const root = new URL("../", import.meta.url);
const html = await readFile(new URL("index.html", root), "utf8");
for (const id of ["why-built", "surprised"]) {
  const tag = html.match(new RegExp(`<section\\b[^>]*id="${id}"[^>]*>`))?.[0];
  assert.ok(
    tag && /\bhidden(?:\s|>)/.test(tag) && /data-placeholder/.test(tag),
    "Reflection placeholders must remain hidden: " + id,
  );
}
const css = await readFile(new URL("css/styles.css", root), "utf8");
assert.match(css, /\[hidden\][^{]*\{\s*display:\s*none\s*!important/);
const fonts = (await readdir(new URL("fonts/", root))).filter((f) =>
  f.endsWith(".woff2"),
);
let total = 0;
for (const file of fonts)
  total += (await stat(new URL("fonts/" + file, root))).size;
assert.ok(total < 150000, "Total WOFF2 budget exceeds 150 KB");
assert.equal(
  (css.match(/@font-face/g) || []).length,
  (css.match(/font-display:\s*swap/g) || []).length,
  "Every font must use swap",
);
console.log(
  `Presentation guards passed; ${fonts.length} Latin WOFF2 subsets, ${total} bytes. Reflection placeholders are hidden.`,
);
