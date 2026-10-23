# Urban Insight

A static city explorer that puts provenance beside every number, built by Brandon Best.

## Data contract

Each metric has a value (or null), unit, source, public source URL, as-of timestamp, kind (live/official/sample) and method note. Missing official values display “Official data not yet loaded”. No arbitrary composites, made-up mode shares, or mixed-unit charts.

The canonical application is this folder in Brandon-Projects-. The portfolio's `public/demos/urban-insight/` is generated with `npm run sync`. Its manifest pins the canonical commit, version and SHA-256 of every public file. Environment files and ingestion scripts are never copied. `npm run check:sync` fails on any difference.

## Run

```sh
npm ci
python3 -m http.server 4180
# Open http://localhost:4180
npm test
npm run lint
npm run contrast
```

Visitors need no build step. Node dependencies are development tools and sources of self-hosted, version-pinned assets.

## Design and accessibility

The Urban Atlas identity pairs Fraunces headings with JetBrains Mono figures/timestamps and Source Sans 3 body copy: an editorial atlas voice with instrument-like data. Newsreader was considered as a quieter serif alternative; Fraunces was chosen for its sturdier, more distinctive display shapes. Mono is not used for section labels or explanatory copy. Four self-hosted Latin WOFF2 subsets total **70,628 bytes**, all using `font-display: swap`; their licenses are in `fonts/`.

Parchment, vellum and umber replace the blue-gray chrome. The map occupies 65vh on desktop, with docked filters and a city panel. Only one map label is shown at a time, placed in screen space to avoid other markers and controls; no city coordinates move. Theme filters apply only to the tile pane, never to attribution, markers or controls. EPA colors remain reserved for AQI. Double keylines separate fills from their backgrounds; all intended text, boundary and interaction pairings are computed in the [full contrast audit](docs/atlas-contrast.md), including raw-fill diagnostics and filtered-tile bounds.

The AQI chart uses a proportional 0–500 segmented EPA ramp. Annotation labels are 200px wide with 12px separation and up to three 90px leader-line lanes. Below 600px, or if a cluster needs more than three lanes, exact values and categories use a stacked list instead. The cluster 51/57/64/83/95 and identical-value cases are unit-tested. Other selected metrics keep a neutral single-unit chart, never an EPA scale. Trend charts show a labeled y-axis, minimum/maximum values, local start/end timestamps and a visible modeled note; exact values remain in the text table.

Phone directory/comparison tables become stacked records with visible source fields. Provenance rules and popover contents are unchanged; headers carry one source badge and rows use compact 44px info targets. Reflection sections remain empty and hidden until Brandon supplies text. CI and the portfolio prebuild check reject exposed placeholders or fonts over 150 KB, and browser checks assert actual computed visibility. See the [Atlas review](ATLAS-REVIEW.md) for before/after screenshots, keyboard and performance results.

Native selectors, arrow-key tabs, visible focus, a skip link, live announcements and text values accompany all graphics. A city table duplicates map selection. Unsupported illustrative congestion, recycling and EV counts are confined to a collapsed disclosure. Invented affordability, energy, water and noise composites are removed.

## Methods

### Official Census snapshot

Set `CENSUS_API_KEY` in this directory's ignored `.env` (see `.env.example`), then run `npm run fetch:acs`. Node 22 loads the environment locally. The script requests six verified tables per city, validates state/place FIPS, strips credential query parameters, checks the serialized output for credentials, and atomically replaces `data/acs.json` only after all cities succeed. It never prints request URLs or raw API errors. With no configured key, it preserves existing output and the empty snapshot renders “Official data not yet loaded”. The deployed snapshot was fetched from the official API, not hand-entered. Do not commit `.env`.

Direct estimates retain published 90% margins of error. Derived shares retain their component estimates and margins; a derived percentage MOE is **not** calculated or implied. Suppressed, negative-sentinel and missing values become null, not zero. The ACS vintage is 2024 (2020–2024 five-year estimates). `B08303` supports the share commuting 60+ minutes, not a fabricated mean commute.

### Refresh and sharing

One Open-Meteo request includes all five coordinates and seven current pollutant fields, plus hourly US AQI. A request aborts after 8 seconds. A successful response is cached in session storage for 15 minutes; Refresh does not bypass that TTL. Failed attempts have a one-minute in-memory cooldown. Storage denial is caught. Partial success retains older valid metrics with stale labels. Fallback is last-good modeled data (stale), then explicitly labeled sample AQI, with no sample number in headlines or charts. Model timestamps older than 90 minutes are also marked stale. Loading regions reserve space; charts expose exact text values.

URLs retain view, selected city, metric, comparison city, category and sort. Share links encode view state, not a frozen data snapshot. CSV exports currently filtered cities, excludes samples, and includes provenance; formula-like text is escaped. A print stylesheet supplies a quieter report. Open-Meteo is free for non-commercial use, not a commercial service-level guarantee.

### Checks

`npm run test:browser` requires the static server on port 4180 and `npx playwright install chromium`. It uses recorded responses and intercepted tiles, never live API requests. `npm run lighthouse` starts its own server, intercepts provider/tile requests and checks three-run mobile medians ≥95 for performance, accessibility, best practices and SEO. GitHub Actions runs unit tests, lint, contrast, browser and Lighthouse checks; screenshots are uploaded as artifacts. Test fixtures are not production data feeds. See `QA.md` for results and `TODO.md` for unresolved legacy-source verification.

The Atlas pass adds `npm run check:presentation`, `npm run test:presentation` (both themes at three widths, contrast probes, responsive tables, hidden-placeholder negative test and resize focus retention) and `npm run test:keyboard` (keyboard-only operation at all three widths). Run `node scripts/check-contrast.mjs --write` to regenerate the full contrast table and `node scripts/capture-design.mjs after` to reproduce the visual review images. The captured before/after comparisons use the same unmodified recorded air response; its older timestamp is honestly marked stale.

CAMS global is a coarse model (~45 km), not a monitor network. City-center estimates can differ from local readings; [AirNow](https://www.airnow.gov/) is linked for official readings. Timestamps are formatted in each city's IANA time zone, including its zone abbreviation.

Census B25077 is an owner-reported home-value estimate, not a sale price. Rent burden is derived by summing B25070's 30–34.9%, 35–39.9%, 40–49.9% and 50%+ bins, divided by the total excluding “not computed”. Commute shares use actual B08301 counts; travel-time distributions use B08303. Survey margins and reference periods differ from modeled air quality.

## Sources and licensing

- [Open-Meteo air-quality documentation](https://open-meteo.com/en/docs/air-quality-api), [terms](https://open-meteo.com/en/terms): free non-commercial endpoint, CAMS attribution, CC BY 4.0.
- [Census ACS 2024 five-year API](https://www.census.gov/data/developers/data-sets/acs-5year.html): 2020–2024 survey estimates; ingestion credentials stay outside public output.
- [OSM tile policy](https://operations.osmfoundation.org/policies/tiles/): visible attribution, normal caching, browser referrer, no prefetch or offline tile downloads.
- Leaflet 1.9.4 and gesture handling 1.2.2 are self-hosted with licenses. Font license is in `fonts/OFL.txt`. Project code inherits the repository MIT license.

## Delivery

Phase 1: provenance, honest data, single-unit charts, ARIA tabs, URL state, accessibility and documentation.
Phase 2: modeled air quality, 15-minute caching, stale/sample fallback, timeout handling and cooperative map interaction.
Phase 3: Census ingestion, comparison, recent-hours trend, CSV/share, portfolio deduplication and QA evidence.
