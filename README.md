<<<<<<< HEAD
# Brandon-Projects-
Data science portfolio bridging e-commerce "street smarts" with economic rigor. Features Python-based arbitrage detectors, inflation trackers, and ML models for fraud and credit risk. Includes NLP sentiment analysis and time-series demand forecasting. View interactive dashboards and deep-dive "Data Stories" for each project. 
=======
# Behavioral Incentives + Policy Impact Analytics Project

This project is tailored to a Los Angeles/USC-focused case study and includes an admissions-ready notebook.

## Selected Case Study 
- City focus: **Los Angeles**
- Policy: **Measure ULA (Mansion Tax)**
- Policy date: **2023-04-01**
- Economic dataset: **FRED series `ATNHPIUS31080Q`** (LA-Long Beach-Anaheim metro all-transactions house price index, quarterly)
- Behavioral sentiment topic: **student loan policy discourse** on Reddit

This setup demonstrates the "human side" of data by pairing expectations/sentiment with measurable market outcomes.

## Project Components
- `src/sentiment_pipeline.py`: Reddit collection + VADER sentiment labeling
- `src/analyze_sentiment.py`: daily sentiment trend + volume chart
- `src/download_fred_series.py`: pulls the chosen FRED economic series
- `src/policy_eda.py`: pre/post policy comparison and trend visualization
- `src/run_project.py`: unified runner with `la-case` and `nyc-case` presets
- `notebooks/admissions_case_study.ipynb`: polished charts + interpretation text for LA

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set your Reddit credentials in `.env`.

## Quick Start (LA/USC Case)
Run everything with one command:
```bash
python src/run_project.py la-case
```

This generates:
- `data/raw/la_hpi_fred.csv`
- `data/processed/reddit_sentiment.csv`
- `reports/policy_trend.png`
- `reports/policy_summary.csv`
- `reports/monthly_series.csv`

Then run sentiment trend aggregation:
```bash
python src/analyze_sentiment.py --input data/processed/reddit_sentiment.csv --output-dir reports
```

## Notebook for Admissions-Ready Output
Open:
- `notebooks/admissions_case_study.ipynb`

The notebook:
- Loads your generated files
- Creates clear visuals
- Auto-generates interpretation text (hypothesis, key findings, incentive-based explanation, and limits)

## Config Files
- `configs/case_study_la_usc.json` (primary)
- `configs/case_study_nyc_hstpa.json` (alternate)

## Optional granular commands
```bash
python src/download_fred_series.py --series-id ATNHPIUS31080Q --output data/raw/la_hpi_fred.csv
python src/sentiment_pipeline.py --query "student loan forgiveness" --subreddit all --limit 300 --output data/processed/reddit_sentiment.csv
python src/policy_eda.py --input data/raw/la_hpi_fred.csv --date-col DATE --value-col ATNHPIUS31080Q --policy-date 2023-04-01 --output-dir reports
python src/analyze_sentiment.py --input data/processed/reddit_sentiment.csv --output-dir reports
```
>>>>>>> 1c2af72 (Initial commit: LA policy + sentiment analytics project)
