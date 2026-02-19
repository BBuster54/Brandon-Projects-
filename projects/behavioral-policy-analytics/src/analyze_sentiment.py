"""Aggregate and visualize sentiment trends from collected social posts."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run(input_path: str, output_dir: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input sentiment file is empty.")

    df["created_utc"] = pd.to_datetime(df["created_utc"])
    daily = (
        df.groupby(df["created_utc"].dt.date, as_index=False)
        .agg(avg_compound=("compound", "mean"), posts=("id", "count"))
        .rename(columns={"created_utc": "date"})
    )

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "sentiment_trend.png")
    daily_path = os.path.join(output_dir, "sentiment_daily.csv")

    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=daily, x="date", y="avg_compound", marker="o", ax=ax1, color="#1f77b4")
    ax1.set_ylabel("Average Compound Sentiment")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    sns.barplot(data=daily, x="date", y="posts", alpha=0.25, ax=ax2, color="#ff7f0e")
    ax2.set_ylabel("Post Count")
    ax2.grid(False)

    plt.title("Daily Sentiment and Discussion Volume")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    daily.to_csv(daily_path, index=False)
    return daily, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved sentiment dataset.")
    parser.add_argument("--input", default="data/processed/sentiment.csv")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    daily, plot_path = run(args.input, args.output_dir)
    print(f"Saved daily summary: {len(daily)} rows")
    print(f"Saved trend plot: {plot_path}")


if __name__ == "__main__":
    main()
