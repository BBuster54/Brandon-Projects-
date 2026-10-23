# Urban Atlas visual identity review

22–23 October 2026. Presentation-only continuation on `codex/urban-insight-v2` in both repositories. Main has not been pushed or merged.

## Before and after

Each pair uses the same unmodified recorded Open-Meteo response and official snapshot, with real OSM tiles. The recorded model timestamp is older and correctly appears as stale. Different viewport/map fitting can load different OSM zoom levels, not different city coordinates. Viewport captures are 1000px high; full-page versions expose tables, ramp and trend.

| Width | Before | After | First-screen detail |
|---|---|---|---|
| 375px | [Full page](docs/screenshots/atlas-before-375.png) | [Full page](docs/screenshots/atlas-after-375.png) | [Before](docs/screenshots/atlas-before-375-viewport.png) / [After](docs/screenshots/atlas-after-375-viewport.png) |
| 768px | [Full page](docs/screenshots/atlas-before-768.png) | [Full page](docs/screenshots/atlas-after-768.png) | [Before](docs/screenshots/atlas-before-768-viewport.png) / [After](docs/screenshots/atlas-after-768-viewport.png) |
| 1440px | [Full page](docs/screenshots/atlas-before-1440.png) | [Full page](docs/screenshots/atlas-after-1440.png) | [Before](docs/screenshots/atlas-before-1440-viewport.png) / [After](docs/screenshots/atlas-after-1440-viewport.png) |
| Dark theme | [Before](docs/screenshots/atlas-before-dark.png) | [After](docs/screenshots/atlas-after-dark.png) | Considered peat/bark/ivory palette; only tiles are filtered |

## Identity and type

Fraunces headings and JetBrains Mono figures/timestamps pair an editorial atlas voice with instrument-like data; Source Sans 3 remains body copy. Newsreader is the alternative: it would be quieter and more reading-oriented, while Fraunces gives this map-led interface a stronger display signature. No all-caps tracked section labels. No authored first-person claims: the two reflection sections have no body copy and remain hidden.

Four Latin WOFF2 files total **70,628 bytes** (under the 150,000-byte budget); all use swap. Headings and body preload only their relevant subsets. No external font/CSP host was added.

## Contrast and map treatment

The [full generated table](docs/atlas-contrast.md) contains every theme-token pairing, normal/hover/focus/disabled filled-button state, raw EPA fill diagnostic, inner fill keyline, outer page/surface keyline, and filtered-tile boundary pairing in both themes. Raw EPA yellow, purple and maroon are not falsely reported as passing by themselves.

- Body text minimum: **6.04:1 light / 8.81:1 dark**, above the prior 5.70 / 8.57 minimums.
- Form control boundaries: **4.45:1 light / 5.83:1 dark**, above the prior 4.28 / 5.44 minimums.
- Filled-button text, normal/focus/disabled: **9.29:1 light / 11.01:1 dark**. Hover: **14.26 / 13.96**. Disabled states retain contrast and use a dashed boundary rather than opacity.
- An inner black/white keyline is selected against each EPA fill (all ≥3:1), with a separate outer black/light-theme or white/dark-theme boundary against its surroundings. Category names remain text in the legend and city list/callouts; samples stay neutral/hatched.
- Tile-filter extremes have conservative analytical bounds and browser canvas probes. Actual darkest light-filter output gives a black boundary **3.44:1**; the brightest dark-filter output gives a white boundary **8.86:1**. Controls/markers/attribution are confirmed unfiltered. Attribution ink on vellum is **14.26:1**.
- Focus remains visible on filled controls, the inverted city card, map controls and map markers. Source popovers remain keyboard-accessible 44px targets.

## Collision rules

The AQI ramp is linear 0–500, with proportional EPA segments and exact-value pins. Values above 500 retain their actual text and receive an explicit off-scale note. Labels are 200px wide, separated by at least 12px, assigned greedily in stable numeric order to at most three 90px lanes. Leader lines run from true positions to labels; positions are not nudged. If the viewport is below 600px or more than three lanes are required, use the stacked exact-value/category list. The cluster **51, 57, 64, 83, 95** and identical values trigger that fallback. Spaced values exercise the staggered case. Sources remain available in both modes.

Map labels use screen-space candidate rectangles scored against other marker hit areas, map bounds and the filter dock. Only the selected label or one hovered/focused label is shown; off-map labels are hidden. Geographic coordinates never change. Resize redraws preserve open chart source details, the hourly table and keyboard focus.

## Verification

- **87 Vitest tests pass**, including all 58 pre-existing tests unchanged. New tests cover clustered/identical/off-scale ramp values, annotation placement, all EPA boundaries and filled states.
- Existing Playwright regression suite passes unchanged: API failure/stale fallback, blocked storage, missing official snapshot, URL restoration, comparison, CSV, ARIA tabs and cooperative touch scrolling.
- New presentation suite passes at **375 / 768 / 1440px in both themes**: no page overflow or clipped source/comparison cells, one visible map label, hidden placeholders (including a negative test that deliberately exposes one and verifies rejection), actual tile-filter probes, filled-button contrast and resize focus retention. The print stylesheet also passes a system-dark contrast check, with placeholders still hidden.
- Keyboard-only operation passes at **all three widths**, using keyboard events without clicks, programmatic focus, selectOption or fill. The pass operates the skip link; Arrow/Home/End tabs; the phone filter disclosure; native type-ahead city/metric/category selectors; reset; Space/Enter map/directory selection with retained focus; comparison and URL state; CSV export; sharing; and the methods disclosure. No map focus trap. Detailed traces and per-width CSV downloads are in ignored `artifacts/atlas-keyboard.json` and `artifacts/atlas-keyboard-*.csv`, uploaded by CI.

Native-select input uses macOS type-ahead and Linux Home/ArrowDown/End keys, with an auto-wait for the same expected value. The first Linux CI run exposed that platform difference in the new test; the key sequence was corrected without changing assertions, app behavior, or any original test.
- ESLint and presentation/font guards pass. The portfolio prebuild check enforces hidden placeholders and the font budget before Next.js builds.
- Final local Lighthouse mobile run, three samples: performance **91 / 100 / 100**, accessibility **100 / 100 / 100**, best practices **100 / 100 / 100**, SEO **100 / 100 / 100**. Medians are **100 in all four categories**, exceeding the unchanged 95 gates. Maximum measured CLS is **0.000244** (not claimed to be zero). An earlier run missed the performance gate; caching equivalent city-local Intl formatters removed repeated constructor work. These are local lab results, not field/SLA claims; CI reruns the same gates.

## Preserved scope and remaining caveats

Compared with commit `0801294`, official/sample data, city metadata, providers, normalization, EPA categories, URL state, exports, recorded fixtures and every original test file are unchanged. The provenance renderer changes only its visual wrapper/summary treatment, not source selection or popover contents. Timestamp formatter reuse produces the same strings. Header/footer anchor destinations and CSP remain unchanged. Portfolio edits unrelated to Urban Insight remain unstaged and untouched.

The original modeled-versus-monitor, ACS uncertainty and legacy sample-source limitations remain documented in `QA.md` and `TODO.md`. The existing 17 npm development-tool advisories were not changed by this visual-only scope. Automated accessibility and keyboard checks are not a substitute for a screen-reader or physical-device audit.
