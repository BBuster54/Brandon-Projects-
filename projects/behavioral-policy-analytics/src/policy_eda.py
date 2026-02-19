"""Policy impact exploratory analysis template.

Expects a CSV with at least:
- date column
- outcome variable (e.g., median_home_price)
- optional policy indicator or a known policy start date
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass
class PolicyEDAConfig:
    input_path: str
    output_dir: str
    date_col: str
    value_col: str
    policy_date: str


def prepare_data(df: pd.DataFrame, date_col: str, value_col: str, policy_date: str) -> pd.DataFrame:
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col)
    data["month"] = data[date_col].dt.to_period("M").dt.to_timestamp()

    monthly = (
        data.groupby("month", as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "monthly_avg_value"})
    )

    policy_ts = pd.to_datetime(policy_date)
    monthly["period"] = monthly["month"].apply(lambda x: "post_policy" if x >= policy_ts else "pre_policy")
    return monthly


def summarize_change(monthly: pd.DataFrame) -> pd.DataFrame:
    summary = monthly.groupby("period", as_index=False)["monthly_avg_value"].mean()
    pre = summary.loc[summary["period"] == "pre_policy", "monthly_avg_value"]
    post = summary.loc[summary["period"] == "post_policy", "monthly_avg_value"]

    if len(pre) == 1 and len(post) == 1 and pre.iloc[0] != 0:
        pct_change = (post.iloc[0] - pre.iloc[0]) / pre.iloc[0] * 100
    else:
        pct_change = float("nan")

    result = pd.DataFrame(
        {
            "metric": ["pre_policy_avg", "post_policy_avg", "percent_change"],
            "value": [pre.iloc[0] if len(pre) else float("nan"), post.iloc[0] if len(post) else float("nan"), pct_change],
        }
    )
    return result


def plot_trend(monthly: pd.DataFrame, policy_date: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "policy_trend.png")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly, x="month", y="monthly_avg_value", marker="o")
    plt.axvline(pd.to_datetime(policy_date), color="red", linestyle="--", label="Policy Start")
    plt.title("Monthly Outcome Trend Around Policy Change")
    plt.xlabel("Month")
    plt.ylabel("Monthly Average")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def run(config: PolicyEDAConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    df = pd.read_csv(config.input_path)
    monthly = prepare_data(df, config.date_col, config.value_col, config.policy_date)
    summary = summarize_change(monthly)
    plot_path = plot_trend(monthly, config.policy_date, config.output_dir)

    monthly_path = os.path.join(config.output_dir, "monthly_series.csv")
    summary_path = os.path.join(config.output_dir, "policy_summary.csv")
    monthly.to_csv(monthly_path, index=False)
    summary.to_csv(summary_path, index=False)

    return monthly, summary, plot_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run policy impact EDA.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output-dir", default="reports", help="Output directory for charts and summary files.")
    parser.add_argument("--date-col", default="date", help="Date column name.")
    parser.add_argument("--value-col", required=True, help="Outcome variable column name.")
    parser.add_argument("--policy-date", required=True, help="Policy start date (YYYY-MM-DD).")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = PolicyEDAConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        date_col=args.date_col,
        value_col=args.value_col,
        policy_date=args.policy_date,
    )
    monthly, summary, plot_path = run(config)
    print(f"Saved monthly series: {len(monthly)} rows")
    print(f"Saved summary metrics: {len(summary)} rows")
    print(f"Saved chart: {plot_path}")


if __name__ == "__main__":
    main()
