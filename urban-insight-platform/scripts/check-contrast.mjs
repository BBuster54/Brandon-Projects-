import { readFile } from "node:fs/promises";
import { contrast } from "../js/contrast.js";
const tokens = JSON.parse(
  await readFile(new URL("../data/theme-tokens.json", import.meta.url)),
);
const css = (
  await readFile(new URL("../css/styles.css", import.meta.url), "utf8")
).toLowerCase();
const rows = [];
for (const [mode, t] of Object.entries(tokens))
  for (const [fg, color] of Object.entries(t)) {
    if (
      !css
        .replace(/#([a-f\d])([a-f\d])([a-f\d])\b/g, "#$1$1$2$2$3$3")
        .includes(color.toLowerCase())
    )
      throw new Error("Token missing from CSS: " + mode + "." + fg);
    for (const bg of ["paper", "surface"]) {
      const ratio = contrast(color, t[bg]),
        required = ["ink", "muted", "accent"].includes(fg)
          ? 4.5
          : fg === "control"
            ? 3
            : null;
      if (required && ratio < required)
        throw new Error("Contrast failed " + mode + " " + fg + "/" + bg);
      rows.push({
        mode,
        foreground: fg,
        background: bg,
        ratio: Number(ratio.toFixed(2)),
        required,
        role:
          fg === "rule"
            ? "decorative only"
            : fg === "control"
              ? "control boundary"
              : required
                ? "text"
                : "background",
      });
    }
  }
console.log(JSON.stringify(rows, null, 2));
