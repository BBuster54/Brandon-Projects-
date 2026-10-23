# Urban Insight v2 — review report

Verified 20 October 2026. Work is on `codex/urban-insight-v2` in both repositories, based on the unmerged Carbon Tracker feature branch. Nothing is pushed to main. Review the Carbon change first, then this stacked change.

## Results

- Vitest: 58 tests passing, including recorded Open-Meteo/ACS fixtures, EPA boundaries, normalization, missing values, cache TTL, timeout, partial fallback, credential stripping, Census geography/bin arithmetic, URL validation and CSV safety.
- ESLint: passing. Portfolio production Next.js/TypeScript build: passing. Scoped portfolio ESLint: passing.
- Playwright: passing at 375, 768 and 1440 pixels, with no page-wide horizontal overflow. Includes API 503 → neutral hatched sample markers, expired cache → stale live-kind values, blocked storage, missing official snapshot, compare mode, restored URL state, CSV download and ARIA arrow-key tabs. Successful requests produce no console or runtime errors.
- Keyboard-only pass: skip link reaches the explorer; native selectors, reset/refresh, map zoom and city markers, directory, sort, comparison, disclosures and exports remain reachable. Space/Enter selects markers and preserves marker focus. Tab traversal is not trapped in the map. Directory selection also preserves focus after rerender. Arrow/Home/End keys move tabs. No pointer interaction is needed for city selection or comparison. `artifacts/keyboard.json` records the traversal.
- Cooperative touch: a simulated one-finger swipe over the map scrolls the page; map gestures use the self-hosted gesture-handling plugin. Two-finger movement and Ctrl/⌘-scroll instructions are visible.
- Lighthouse CI, mobile, three runs with recorded provider responses and intercepted tiles: performance **95 / 100 / 100**, accessibility **100 / 100 / 100**, best practices **100 / 100 / 100**, SEO **100 / 100 / 100**. All medians are **100**. Measured CLS is **0** in these three runs. These are local lab results, not a claim about field performance or provider availability. CI reruns the same ≥95 median gates.
- Live smoke: real Open-Meteo and OSM tiles loaded, official ACS snapshot loaded; no runtime errors. Print-to-PDF generated successfully.
- Contrast: every theme token's intended pairing is computed by `npm run contrast`. Minimum text contrast is 5.70:1 light / 8.57:1 dark. Control boundaries are ≥4.28:1 light / ≥5.44:1 dark. Pale rules are decorative only. EPA colors appear as outlined dots with readable text labels, never as body text.
- Security: environment file remains ignored. Exact local credential is excluded from generated data and public code; source URLs strip key/token parameters. CSP permits only the specific Open-Meteo API and OSM tile hosts; scripts/fonts are local. Obsolete public assets and misleading sample credits are removed (recoverable through Git history).

## Screenshots

Captured with real providers from the local static app. Recorded-fixture failure/dark screenshots, CSV, keyboard trace and PDF are also produced under ignored `artifacts/`; GitHub Actions uploads QA artifacts.

| Width | Full-page screenshot |
| --- | --- |
| 375px | [Mobile](docs/screenshots/urban-375.png) |
| 768px | [Tablet](docs/screenshots/urban-768.png) |
| 1440px | [Desktop](docs/screenshots/urban-1440.png) |

## Every remaining sample figure

All are unverified legacy illustrative values, not current observations. Congestion, recycling and EV counts appear **only** in the collapsed “Illustrative sample indicators” disclosure, each with a **Source pending** badge. AQI sample numbers are retained in the source file only as the final fallback's provenance; the UI shows a dash and “No current reading”, never that number or an EPA category color. Samples are excluded from rankings, comparisons, trends and CSV exports.

| City | Legacy AQI (not displayed) | Illustrative congestion % | Illustrative recycling % | Illustrative EV count |
| --- | ---: | ---: | ---: | ---: |
| New York City | 89 | 78 | 45 | 485 |
| Los Angeles | 124 | 89 | 38 | 658 |
| Chicago | 76 | 65 | 52 | 312 |
| Houston | 98 | 72 | 35 | 245 |
| Phoenix | 105 | 58 | 28 | 198 |

## Known-problem changelog

| Original problem | Resolution |
| --- | --- |
| 1. Fabricated commute shares / hardcoded Other | Actual B08301 counts and denominators; Other sums taxi, motorcycle and other-mode counts. B08303 uses the actual 60+ minute bins, not a fabricated mean. |
| 2. Mixed units / construction divided by two | Single selected metric and unit per ranking; comparison rows also preserve units. Removed arbitrary scaling and composites. |
| 3. Misleading sample source credits | Explicit per-metric provenance; legacy values say illustrative / Source pending, with no fictitious official source attribution. |
| 4. No map / ineffective reset / incorrect real-time claim | Pinned self-hosted Leaflet map, accessible city directory, AQI category filtering, complete state reset and accurate portfolio copy. Exactly one Urban card. |
| 5. Inaccessible tabs/charts / no URL or freshness | Real ARIA tabs, keyboard controls, query state, same-unit rankings, trend table, local time zones, timeout/cache/stale/sample states and polite announcements. |
| 6. Unpinned CDN / unused adapter and CSS | Removed Chart.js and its adapter; zero chart dependencies, self-hosted Leaflet 1.9.4, gesture handling 1.2.2 and licensed Source Sans 3. Removed unused old public files. |
| 7. Undated sample data / mixed vintages | Samples quarantined; modeled as-of dates and ACS 2020–2024 survey periods/retrieval dates visible. No invented replacements. |

## Limitations and follow-up

CAMS global (~45km) is not a local monitor network; use the linked AirNow service for official readings. ACS home values are owner-reported, not market prices. Derived share margins are not calculated; component published 90% margins are retained. The free air endpoint has no commercial SLA. Share URLs restore view state, not an immutable data snapshot. Small screens can use the directory or zoom to separate nearby city markers.

`npm audit` reports 17 development-tool advisories (including Lighthouse CI's transitive dependencies); no npm dependency is shipped as a runtime application bundle. Do not describe the development toolchain as audit-clean. A breaking toolchain upgrade requires a separate compatibility review; see `TODO.md`. Automated accessibility and this keyboard pass do not replace a screen-reader or real-device audit.
