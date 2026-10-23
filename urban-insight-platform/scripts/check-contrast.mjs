import { readFile, writeFile } from "node:fs/promises";
import { contrast } from "../js/contrast.js";
import { categories } from "../js/aqi.js";
import { fillEdge, tileRange } from "../js/render/epa-style.js";
const tokens = JSON.parse(
  await readFile(new URL("../data/theme-tokens.json", import.meta.url)),
);
const css = (
  await readFile(new URL("../css/styles.css", import.meta.url), "utf8")
)
  .toLowerCase()
  .replace(/#([a-f\d])([a-f\d])([a-f\d])\b/g, "#$1$1$2$2$3$3");
const rows = [];
function record(mode, pair, fg, bg, required = null, note = "") {
  const ratio = contrast(fg, bg);
  rows.push({
    mode,
    pair,
    foreground: fg,
    background: bg,
    ratio: Number(ratio.toFixed(2)),
    required,
    note,
  });
  if (required && ratio < required)
    throw new Error(
      `Contrast failed: ${mode} ${pair} ${ratio.toFixed(2)} < ${required}`,
    );
}
for (const [mode, t] of Object.entries(tokens)) {
  for (const [name, color] of Object.entries(t)) {
    if (!css.includes(color.toLowerCase()))
      throw new Error("Missing CSS token " + mode + "." + name);
    for (const bg of ["paper", "surface"]) {
      const text = ["ink", "muted", "accent"].includes(name);
      const required = text
        ? mode === "light"
          ? 5.7
          : 8.57
        : ["control", "edge"].includes(name)
          ? name === "control"
            ? mode === "light"
              ? 4.28
              : 5.44
            : 3
          : null;
      record(
        mode,
        `${name} / ${bg}`,
        color,
        t[bg],
        required,
        required
          ? "Intended text or boundary pairing"
          : "Background/context only; not a text pairing",
      );
    }
  }
  for (const [state, fill] of Object.entries({
    normal: t.accent,
    hover: t.ink,
    disabled: t.accent,
    focus: t.accent,
  })) {
    record(
      mode,
      `Filled button ${state}: text / fill`,
      t.buttonText,
      fill,
      mode === "light" ? 5.7 : 8.57,
    );
    for (const bg of ["paper", "surface"])
      record(mode, `Filled button ${state}: boundary / ${bg}`, fill, t[bg], 3);
  }
  record(mode, "Focus ring / surface", t.ink, t.surface, 3);
  record(mode, "Focus ring / page", t.ink, t.paper, 3);
  record(
    mode,
    "Attribution text / unfiltered vellum",
    "#2b251c",
    "#fff7e9",
    5.7,
  );
  for (const [key, pixel] of Object.entries(tileRange[mode]))
    record(
      mode,
      `Map controls/focus boundary / filtered tile ${key}`,
      t.edge,
      pixel,
      3,
    );
  record(
    mode,
    "Inverted card text / fill",
    t.paper,
    t.ink,
    mode === "light" ? 5.7 : 8.57,
  );
  for (const c of categories) {
    for (const bg of ["paper", "surface"])
      record(
        mode,
        `${c.label}: raw fill / ${bg}`,
        c.color,
        t[bg],
        null,
        "Diagnostic: fill alone may fail; not the boundary treatment",
      );
    record(
      mode,
      `${c.label}: inner keyline / fill`,
      fillEdge(c.color),
      c.color,
      3,
    );
    for (const bg of ["paper", "surface"])
      record(mode, `${c.label}: outer keyline / ${bg}`, t.edge, t[bg], 3);
    for (const [key, pixel] of Object.entries(tileRange[mode]))
      record(
        mode,
        `${c.label}: map outer boundary / filtered tile ${key}`,
        t.edge,
        pixel,
        3,
        "Tile-filter output bound, covers every original sRGB pixel",
      );
  }
}
console.log(JSON.stringify(rows, null, 2));
if (process.argv.includes("--write"))
  await writeFile(
    new URL("../docs/atlas-contrast.md", import.meta.url),
    "# Urban Atlas contrast audit\n\nComputed WCAG sRGB relative-luminance ratios. Raw EPA fills are diagnostic, not standalone accessible boundaries. A 2px inner black/white keyline contrasts against the fill; a separate outer black (light) / white (dark) boundary contrasts against the surface or filtered tile. Category names remain text. Focus rings use a surface gap. Disabled filled buttons keep their contrast and use a dashed boundary, not opacity.\n\nLight tile filter: grayscale(1) sepia(.3) brightness(.8) contrast(.4) brightness(1.3), ideal channel range 0.39–0.806. Dark filter: grayscale(1) sepia(.25) invert(1) contrast(.3) brightness(.45), ideal channel range 0.1575–0.2925. We pad the 8-bit extrema by three code values for rounding: #606060–#D0D0D0 light, #262626–#4D4D4D dark. The filter applies only to .leaflet-tile-pane; UI/attribution/markers are not filtered. The grayscale-first filter makes the extrema independent of original tile colors. Browser canvas probes of the same filters are also tested: light black input produces #626262 (boundary 3.44:1), dark black input #4A4A4A (8.86:1).\n\n| Theme | Pairing | Foreground | Background | Ratio | Minimum | Note |\n|---|---|---|---|---:|---:|---|\n" +
      rows
        .map(
          (r) =>
            `| ${r.mode} | ${r.pair} | ${r.foreground} | ${r.background} | ${r.ratio.toFixed(2)} | ${r.required ?? "—"} | ${r.note} |`,
        )
        .join("\n") +
      "\n",
  );
