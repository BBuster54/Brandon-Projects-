# Brandon Projects

Data science portfolio bridging practical market intuition with economic rigor. This repo includes policy analytics, NLP sentiment systems, causal impact estimation, and forecasting workflows.

<<<<<<< HEAD
## Selected Case Study 
- City focus: **Los Angeles**
- Policy: **Measure ULA (Mansion Tax)**
- Policy date: **2023-04-01**
- Economic dataset: **FRED series `ATNHPIUS31080Q`** (LA-Long Beach-Anaheim metro all-transactions house price index, quarterly)
- Behavioral sentiment topic: **student loan policy discourse** on Reddit
=======
## Flagship Project: Behavioral Incentives + Policy Impact Analytics
>>>>>>> 947a52b (Add GDELT no-key sentiment pipeline, dashboard upgrades, and gitignore)

This project is a comparative policy research platform centered on behavioral economics.

### Core Questions
- How do policy shocks change housing market trajectories?
- Can social sentiment act as a lead indicator for market movement?
- How do LA and NYC differ in policy response dynamics?

### Case Studies
- Los Angeles: Measure ULA (policy date: 2023-04-01)
- New York City: HSTPA Rent Reform (policy date: 2019-06-14)

Economic data source:
- FRED `ATNHPIUS31080Q` (LA metro house price index)
- FRED `ATNHPIUS35620Q` (NYC metro house price index)

Sentiment data source:
- **Default (no key):** GDELT DOC API
- Optional: Reddit via PRAW API

## High-Impact Additions Implemented

### 1. Causal Impact Analysis
- Interrupted time series counterfactual modeling
- Estimated treatment effects post-policy
- Confidence intervals and observed-vs-counterfactual chart
- File: `src/causal_impact.py`

### 2. Cross-City Comparison (LA vs NYC)
- Policy effectiveness comparison
- Market divergence visualization
- Optional differential sentiment comparison
- File: `src/cross_city_compare.py`

### 3. Lagged Sentiment -> Market Prediction
- Lagged linear models across windows
- Granger causality tests
- Best-lag selection with R2/RMSE outputs
- File: `src/lagged_prediction.py`

### 4. Topic Modeling Layer (NLP Depth)
- LDA topic modeling over social/news text
- Topic keyword extraction
- Topic evolution over time
- File: `src/topic_modeling.py`

### 5. Mandatory Interactive Dashboard
- Streamlit dashboard with city dropdown
- Policy-date slider for scenario simulation
- Sentiment vs price overlays
- Counterfactual and cross-city summary panels
- File: `dashboard/app.py`

## Main Workflows
- `src/run_project.py` contains all workflows.

Key commands:
```bash
python src/run_project.py la-case
python src/run_project.py nyc-case
python src/run_project.py full-platform
python src/run_project.py compare-cities
python src/run_project.py causal --treated-input data/raw/la_hpi_fred.csv --value-col ATNHPIUS31080Q --policy-date 2023-04-01
python src/run_project.py predict-lags --monthly-series-input reports/la/monthly_series.csv --sentiment-daily-input reports/la/sentiment_daily.csv
python src/run_project.py topics --input data/processed/la_sentiment.csv --output-dir reports/la/topics
```

Use Reddit instead of GDELT:
```bash
python src/run_project.py la-case --sentiment-source reddit
```

## Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard reads outputs under:
- `reports/la/`
- `reports/nyc/`
- `reports/comparison/`

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

No API keys are required for the default GDELT pipeline.

## Repo Structure
- `src/`: analytics pipelines
- `dashboard/`: interactive app
- `configs/`: case definitions
- `data/raw/`: downloaded source datasets
- `data/processed/`: NLP outputs
- `reports/la/`, `reports/nyc/`, `reports/comparison/`: generated deliverables
- `notebooks/`: admissions-ready narrative notebook
