# Carbon Footprint Tracker

A Python command-line tool that estimates a household’s monthly carbon footprint across transportation, electricity, home heating, air travel, and waste. It prints a category breakdown, exports a CSV summary, and generates a Matplotlib chart.

## Run

```bash
python3 emissions_calculator.py
```

Use flags to model a different month:

```bash
python3 emissions_calculator.py --transport train --commute-miles 350 --electricity-kwh 500 --flight-miles 250
```

The chart and CSV are written to `output/`.

## Method note

This is an educational prototype. Its emission factors are illustrative estimates, documented in the source code, and should not be used for formal carbon accounting.

## Technologies

Python, Matplotlib, CSV, and the standard library.
