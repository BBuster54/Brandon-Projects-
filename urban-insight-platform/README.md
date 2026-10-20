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

Source Sans 3 is self-hosted for compact, readable civic data and tabular numerals. Cool paper, ink and graphite leave EPA colors for AQI only. No invented red/green judgment for housing or infrastructure. Contrast is calculated in `scripts/check-contrast.mjs`; control borders exceed 3:1 and text exceeds 4.5:1 in both themes. Pale rule borders are decorative only.

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
