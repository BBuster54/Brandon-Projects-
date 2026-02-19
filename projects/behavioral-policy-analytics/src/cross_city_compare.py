"""Cross-city policy comparison (LA vs NYC) for policy metrics and sentiment response."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass
class CompareConfig:
    la_policy_summary: str
    nyc_policy_summary: str
    la_monthly_series: str
    nyc_monthly_series: str
    la_sentiment_file: str | None
    nyc_sentiment_file: str | None
    output_dir: str


def _extract(summary_path: str) -> dict:
    df = pd.read_csv(summary_path)
    out = {r["metric"]: r["value"] for _, r in df.iterrows()}
    return out


def run(config: CompareConfig) -> tuple[pd.DataFrame, str]:
    os.makedirs(config.output_dir, exist_ok=True)

    la = _extract(config.la_policy_summary)
    nyc = _extract(config.nyc_policy_summary)

    comp = pd.DataFrame(
        {
            "city": ["Los Angeles", "New York City"],
            "pre_policy_avg": [la.get("pre_policy_avg"), nyc.get("pre_policy_avg")],
            "post_policy_avg": [la.get("post_policy_avg"), nyc.get("post_policy_avg")],
            "percent_change": [la.get("percent_change"), nyc.get("percent_change")],
        }
    )

    if config.la_sentiment_file and config.nyc_sentiment_file:
        la_sent = pd.read_csv(config.la_sentiment_file)
        nyc_sent = pd.read_csv(config.nyc_sentiment_file)
        comp["avg_sentiment"] = [la_sent["compound"].mean(), nyc_sent["compound"].mean()]
        comp["posts"] = [len(la_sent), len(nyc_sent)]

    comp["effectiveness_rank"] = comp["percent_change"].rank(ascending=False, method="dense")

    out_csv = os.path.join(config.output_dir, "cross_city_comparison.csv")
    out_plot = os.path.join(config.output_dir, "cross_city_divergence.png")
    comp.to_csv(out_csv, index=False)

    la_monthly = pd.read_csv(config.la_monthly_series)
    nyc_monthly = pd.read_csv(config.nyc_monthly_series)
    la_monthly["city"] = "Los Angeles"
    nyc_monthly["city"] = "New York City"

    merged = pd.concat([la_monthly, nyc_monthly], ignore_index=True)
    merged["month"] = pd.to_datetime(merged["month"])

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 5))
    sns.lineplot(data=merged, x="month", y="monthly_avg_value", hue="city")
    plt.title("Housing Market Divergence: LA vs NYC")
    plt.xlabel("Month")
    plt.ylabel("Average Index")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    plt.close()

    return comp, out_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare policy outcomes across cities.")
    parser.add_argument("--la-policy-summary", required=True)
    parser.add_argument("--nyc-policy-summary", required=True)
    parser.add_argument("--la-monthly-series", required=True)
    parser.add_argument("--nyc-monthly-series", required=True)
    parser.add_argument("--la-sentiment-file")
    parser.add_argument("--nyc-sentiment-file")
    parser.add_argument("--output-dir", default="reports/comparison")
    args = parser.parse_args()

    cfg = CompareConfig(
        la_policy_summary=args.la_policy_summary,
        nyc_policy_summary=args.nyc_policy_summary,
        la_monthly_series=args.la_monthly_series,
        nyc_monthly_series=args.nyc_monthly_series,
        la_sentiment_file=args.la_sentiment_file,
        nyc_sentiment_file=args.nyc_sentiment_file,
        output_dir=args.output_dir,
    )
    comp, plot_path = run(cfg)
    print(comp)
    print(f"Saved divergence plot: {plot_path}")


if __name__ == "__main__":
    main()
