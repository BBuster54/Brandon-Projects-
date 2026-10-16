# Carbon Footprint Tracker
A transparent, live calculator that helps you understand six parts of your monthly footprint and explore practical changes.

[Live demo](https://final-portfolio-website.vercel.app/demos/carbon-footprint-tracker/) · [Portfolio](https://final-portfolio-website.vercel.app/projects)

## Interface
IBM Plex Sans, ink blue, cool white and coral, with a stacked footprint bar as the focal point. Self-hosted fonts, no runtime dependencies, responsive results-first mobile layout, native form controls and text alternatives for visual data.
Screenshots and measured QA results are recorded in `QA.md` when validation is complete.

## Model
This repository's `data/factors.json` is canonical. Python and JavaScript load the same file. The portfolio holds a generated copy plus a SHA-256/version/revision manifest. Run `npm run sync` after committing a canonical phase; `npm run check:sync` fails on differing app files or factor versions/checksums. Set `PORTFOLIO_ROOT` if the portfolio is not a sibling checkout. Never edit the generated copy directly.

All quantities are monthly except weekly commute inputs (× 52/12). Kilometers divide by 1.609344. Diet multiplies daily factors by 30. Annual projections multiply monthly results by 12. Calculations retain full precision until display. Default household size is 1.

Transport, flights and diet are per-person inputs. Electricity, natural gas and waste are whole-household inputs. One-person results divide shared categories by household size. Whole-household results multiply personal categories by household size, assuming identical habits.

Confidence bands: high ±10%, medium ±25%, low ±50%. Total bounds **sum category bounds** rather than combining by quadrature; independence of factor errors is not established. These are sensitivity assumptions, not statistical confidence intervals. All current factors have low confidence pending verification.

## Factor sources
| Factor | kg CO₂e per unit | Source status |
| --- | --- | --- |
| Car | 0.404 / mile | Pending |
| Bus | 0.14 / passenger mile | Pending |
| Train | 0.09 / passenger mile | Pending |
| Bike / walk | 0 / mile | Direct operational boundary; pending |
| Electricity | 0.371 / kWh | Pending |
| Natural gas | 5.3 / therm | Pending |
| Flights | 0.255 / passenger mile | Pending |
| Waste | 12 / bag | Pending; bag mass unspecified |
| Diet: meat / mixed / vegetarian / vegan | 7 / 5 / 3.8 / 2.9 / person / day | Illustrative, unsourced |

The UI methodology table is generated from JSON. See [source verification tasks](TODO-SOURCES.md). No benchmark is presented as sourced without verification.

## Run
The web app is static and build-free. Serve it over HTTP (JSON and modules cannot reliably load from `file://`):
```sh
python3 -m http.server 4173
# Open http://localhost:4173
```
Python:
```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python emissions_calculator.py --commute-miles-per-week 150 --diet vegetarian
.venv/bin/python emissions_calculator.py --no-chart
```
`--commute-miles` and `--commute-miles-per-week` are mutually exclusive. Matplotlib is a third-party chart dependency; calculations and CSV export work without it using `--no-chart`.

## Tests and automation
```sh
npm ci
npm test
.venv/bin/python -m pytest tests/test_model.py -q
npm run sync
npm run check:sync
npm run lighthouse
```
Vitest and pytest use shared known-answer fixtures, including 600 car miles = 242.4 kg and a train swap saving 188.4 kg, weekly conversion, kilometers and household allocation. GitHub Actions runs both suites and mobile Lighthouse CI with 95-point minimums for performance, accessibility, best practices and SEO. Lighthouse uses a local HTTP server; preview deployments enforce security response headers.

## Deployment
For a standalone Vercel preview, set root directory to `carbon-footprint-tracker`, framework to Other, no build command and output directory `.`. For the existing portfolio deployment, run the sync script, commit its output on the portfolio feature branch, and use the Git integration's preview. Review both branches before merging. No production deploy is needed for review.

The app includes a restrictive CSP meta policy. `vercel.json` sets CSP, frame restrictions, MIME sniffing prevention, referrer and permissions policies. The portfolio applies the same headers only to this demo.

## Limitations and roadmap
Educational estimates only, with unverified, heterogeneous factor boundaries. No live utility data, verified lifecycle accounting or comprehensive personal inventory. Bike and walking zeros exclude food and manufacturing; flight altitude effects, regional grids and waste composition are not modeled. Verify factors and their boundaries before considering benchmarks.

Phase 1: unified model, live calculation, validation, results, uncertainty, accessibility, methodology, tests and documentation.
Phase 2: comparison scenarios and monthly goals.
Phase 3: local history, sharing, exports and print reports.

## License
MIT, inherited from [the repository license](../LICENSE). IBM Plex Sans is distributed under the SIL Open Font License; see `fonts/OFL.txt`.

