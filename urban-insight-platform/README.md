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
