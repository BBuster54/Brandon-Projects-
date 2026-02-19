"""Lagged sentiment -> market prediction with regression and Granger tests."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import grangercausalitytests


@dataclass
class LagPredictionConfig:
    monthly_series_input: str
    sentiment_daily_input: str
    output_dir: str
    max_lag: int


def _prepare(monthly_path: str, sentiment_path: str, max_lag: int) -> pd.DataFrame:
    housing = pd.read_csv(monthly_path)
    housing["month"] = pd.to_datetime(housing["month"])
    housing = housing[["month", "monthly_avg_value"]].copy()

    sentiment = pd.read_csv(sentiment_path)
    if "date" in sentiment.columns:
        sentiment["date"] = pd.to_datetime(sentiment["date"])
    else:
        sentiment["date"] = pd.to_datetime(sentiment["created_utc"])

    sentiment["month"] = sentiment["date"].dt.to_period("M").dt.to_timestamp()
    sent_monthly = sentiment.groupby("month", as_index=False)["avg_compound"].mean()

    data = housing.merge(sent_monthly, on="month", how="inner").sort_values("month")
    for lag in range(1, max_lag + 1):
        data[f"sent_lag_{lag}"] = data["avg_compound"].shift(lag)

    data = data.dropna().reset_index(drop=True)
    return data


def run(config: LagPredictionConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    os.makedirs(config.output_dir, exist_ok=True)
    data = _prepare(config.monthly_series_input, config.sentiment_daily_input, config.max_lag)

    lag_metrics = []
    best_lag = None
    best_r2 = -np.inf

    for lag in range(1, config.max_lag + 1):
        feature = [f"sent_lag_{lag}"]
        df = data[["month", "monthly_avg_value"] + feature].dropna().copy()
        if len(df) < 8:
            continue

        split = max(4, int(len(df) * 0.8))
        train = df.iloc[:split]
        test = df.iloc[split:]
        if len(test) == 0:
            continue

        model = LinearRegression()
        model.fit(train[feature], train["monthly_avg_value"])
        pred = model.predict(test[feature])

        r2 = r2_score(test["monthly_avg_value"], pred)
        rmse = float(mean_squared_error(test["monthly_avg_value"], pred) ** 0.5)

        lag_metrics.append({"lag": lag, "r2": float(r2), "rmse": rmse})
        if r2 > best_r2:
            best_r2 = r2
            best_lag = lag

    lag_df = pd.DataFrame(lag_metrics)
    lag_path = os.path.join(config.output_dir, "lag_model_metrics.csv")
    lag_df.to_csv(lag_path, index=False)

    if best_lag is None:
        raise ValueError("Insufficient data for lag modeling.")

    granger_input = data[["monthly_avg_value", "avg_compound"]].dropna()
    granger = grangercausalitytests(granger_input, maxlag=config.max_lag, verbose=False)
    granger_rows = []
    for lag, results in granger.items():
        pval = float(results[0]["ssr_ftest"][1])
        granger_rows.append({"lag": lag, "ssr_ftest_pvalue": pval})

    granger_df = pd.DataFrame(granger_rows)
    granger_path = os.path.join(config.output_dir, "granger_results.csv")
    granger_df.to_csv(granger_path, index=False)

    best_data = data[["month", "monthly_avg_value", f"sent_lag_{best_lag}"]].dropna().copy()
    model = LinearRegression().fit(best_data[[f"sent_lag_{best_lag}"]], best_data["monthly_avg_value"])
    best_data["predicted"] = model.predict(best_data[[f"sent_lag_{best_lag}"]])

    plot_path = os.path.join(config.output_dir, "lag_prediction_fit.png")
    plt.figure(figsize=(10, 5))
    plt.plot(best_data["month"], best_data["monthly_avg_value"], label="Actual")
    plt.plot(best_data["month"], best_data["predicted"], label=f"Predicted (lag={best_lag})", linestyle="--")
    plt.title("Lagged Sentiment Prediction Fit")
    plt.xlabel("Month")
    plt.ylabel("Housing Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    summary = pd.DataFrame(
        {
            "metric": ["best_lag", "best_lag_r2", "best_lag_rmse"],
            "value": [
                int(best_lag),
                float(lag_df.loc[lag_df["lag"] == best_lag, "r2"].iloc[0]),
                float(lag_df.loc[lag_df["lag"] == best_lag, "rmse"].iloc[0]),
            ],
        }
    )
    summary_path = os.path.join(config.output_dir, "lag_prediction_summary.csv")
    summary.to_csv(summary_path, index=False)

    return lag_df, granger_df, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lagged sentiment prediction modeling.")
    parser.add_argument("--monthly-series-input", required=True)
    parser.add_argument("--sentiment-daily-input", required=True)
    parser.add_argument("--output-dir", default="reports/prediction")
    parser.add_argument("--max-lag", type=int, default=6)
    args = parser.parse_args()

    cfg = LagPredictionConfig(
        monthly_series_input=args.monthly_series_input,
        sentiment_daily_input=args.sentiment_daily_input,
        output_dir=args.output_dir,
        max_lag=args.max_lag,
    )
    lag_df, granger_df, plot_path = run(cfg)
    print(lag_df)
    print(granger_df)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
