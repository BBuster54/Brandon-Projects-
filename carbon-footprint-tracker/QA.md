# Carbon calculator QA and delivery notes

Initial visual and Lighthouse checks ran October 16, 2026; final unit, Python, interaction and keyboard checks were repeated October 20, 2026. Changes are on `codex/carbon-tracker-v2` in both repositories. Review preview deployments before merging; production branches were not changed.

## Automated checks

- Vitest: 30 passing tests for known answers, conversion, validation, uncertainty, scenarios, sharing, history and export metadata.
- pytest: 13 passing tests using the same JSON fixtures, including the weekly CLI argument and mutually exclusive commute options.
- Chromium interaction checks: default and changed results, unit conversion without changing emissions, blank/high-value feedback, scope-safe scenarios, goals, monthly snapshots, deletion, URL restoration, CSV/JSON/PNG downloads, clipboard fallback, blocked storage and no console/runtime errors.
- Firefox 142.0.1 and WebKit 26.0 smoke checks: calculation, presets, persisted history, URL restoration and invalid input.
- Mobile Lighthouse: Performance 100, Accessibility 100, Best Practices 100, SEO 100; CLS approximately 0.002. These are measured local lab scores, not a guarantee of scores on every device/network.
- Printed report: one A4 page in Chromium, with source status, model version, input summary, bounds and optional comparison totals.
- Portfolio: lint and production build, plus an exact canonical factor revision/version/SHA-256 check.
- GitHub Actions: JavaScript, Python, browser interaction tests and a Lighthouse CI minimum of 95 in all four categories. Portfolio CI independently verifies its copied factors.

## Screenshots

- [375 px](docs/screenshots/carbon-375.png)
- [768 px](docs/screenshots/carbon-768.png)
- [1440 px](docs/screenshots/carbon-1440.png)

All three were inspected and checked for horizontal overflow. Mobile places results before inputs; desktop uses inputs beside the results. Charts have visible text alternatives: category list, comparison table and monthly snapshot list.

## Keyboard-only pass

The automated journey uses Tab, Shift+Tab, Enter and keyboard selection, without mouse clicks or programmatic focus during the journey.
It covers the skip link, household field, form submission, travel-mode selection, baseline capture, saving a snapshot, copying a link (including clipboard-denied fallback), all export controls, and opening the methodology disclosure.
Visible focus is asserted at every visited control, including the native month input's internal segments.
Clipboard fallback focuses a selectable link. Shift+Tab returns to export controls. Clear-history confirmation and deletion restore focus to useful controls.

## Manual review checklist

- [x] Inspect 375, 768 and 1440 px screenshots and check overflow.
- [x] Verify direct labels remain legible independently of chart colors.
- [x] Check light and dark theme rendering.
- [x] Confirm invalid fields show messages and prevent stale export/snapshot actions.
- [x] Verify storage-blocked behavior.
- [x] Download CSV, JSON and PNG; generate a one-page print report.
- [x] Exercise keyboard navigation and native form behavior.
- [ ] Review on a physical iPhone and current Safari/Edge releases. Local WebKit is the frozen build available on this Mac; the engine test is not a physical-device certification.
- [ ] Verify original emission-factor sources and select a benchmark with matching scope.

## Known-problem changelog

| Original issue                                             | Resolution                                                                                                                                              |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Transport factors about 10× too high; electricity mismatch | Both engines load the inherited Python values from one canonical JSON model. Known-answer tests enforce them.                                           |
| CLI and web model different categories                     | Both cover transportation, electricity, natural gas, flights, waste and diet.                                                                           |
| Inert Enter key; blank becomes zero; negative input        | Native form, 100 ms live updates, Enter handling, inline validation, and high-usage warnings.                                                           |
| Unpinned Chart.js CDN                                      | Removed. CSS/SVG charts and self-hosted licensed fonts have no runtime CDN dependency.                                                                  |
| Canvas has no alternative; results not announced           | Visible category list, comparison table and history list, with a polite results announcement.                                                           |
| Unformatted totals and wrong unit                          | Shared Intl formatter, CO₂e units, annual projection, metric tons, uncertainty bounds, and user goals. Benchmarks remain disabled pending verification. |
| Narrow stack and colliding export button                   | Responsive two-column layout, stacked hero bar, consistent action spacing and dedicated export section.                                                 |
| Missing navigation, metadata and icon                      | Portfolio/source links, description, canonical/Open Graph/Twitter metadata and SVG favicon.                                                             |
| Unsourced drifting methodology                             | Methodology generated from shared JSON; Source pending badges; verification queue in TODO-SOURCES.md.                                                   |

## Remaining limits

The model remains educational: inherited factors have no verified citations and use different system boundaries. All currently use the low-confidence ±50% assumption. Bounds are summed, not combined by quadrature; they are not statistical intervals. Diet uses 30 days and annual figures are monthly ×12. Household totals assume identical personal habits.

The static application ships no npm runtime dependencies. The requested Lighthouse CI development toolchain currently reports npm audit findings in transitive packages (17 findings after a nonbreaking audit fix). These packages are used for local/CI auditing, not served to visitors. Do not treat this as a clean audit of the development toolchain.

## Vercel review and merge

1. Open the portfolio feature-branch preview, using `/demos/carbon-footprint-tracker/index.html`.
2. Review inputs, methodology, scenarios, history and exported reports.
3. Review the three phase commits in each repository. The portfolio manifest pins the exact canonical revision.
4. Merge the canonical project PR, then the portfolio PR after checks pass. The portfolio's existing Vercel integration will deploy production after that merge.
5. For later model changes, commit in Brandon-Projects first, run `npm run sync`, then commit the generated portfolio copy. Never edit its factors independently.
