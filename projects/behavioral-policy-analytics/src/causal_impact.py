"""Quasi-causal policy impact estimation with counterfactual trajectories.

Implements an interrupted time series model with optional control series.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class CausalConfig:
    treated_input: str
    date_col: str
    value_col: str
    policy_date: str
    output_dir: str
    control_input: str | None = None
    control_value_col: str | None = None


def _prepare_single_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    data = df[[date_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).dropna()
    data["month"] = data[date_col].dt.to_period("M").dt.to_timestamp()
    return data.groupby("month", as_index=False)[value_col].mean()


def run(config: CausalConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    os.makedirs(config.output_dir, exist_ok=True)
    policy_ts = pd.to_datetime(config.policy_date)

    treated_raw = pd.read_csv(config.treated_input)
    treated = _prepare_single_series(treated_raw, config.date_col, config.value_col).rename(
        columns={config.value_col: "y"}
    )
    treated["t"] = np.arange(len(treated))
    treated["post"] = (treated["month"] >= policy_ts).astype(int)
    treated["t_post"] = treated["t"] * treated["post"]

    if config.control_input and config.control_value_col:
        control_raw = pd.read_csv(config.control_input)
        control = _prepare_single_series(control_raw, config.date_col, config.control_value_col).rename(
            columns={config.control_value_col: "control"}
        )
        data = treated.merge(control, on="month", how="inner")
        model = smf.ols("y ~ t + post + t_post + control", data=data).fit()
    else:
        data = treated.copy()
        model = smf.ols("y ~ t + post + t_post", data=data).fit()

    # Counterfactual enforces no treatment: post=0 and t_post=0.
    cf_data = data.copy()
    cf_data["post"] = 0
    cf_data["t_post"] = 0

    pred = model.get_prediction(cf_data).summary_frame(alpha=0.05)
    data["counterfactual"] = pred["mean"]
    data["cf_ci_low"] = pred["mean_ci_lower"]
    data["cf_ci_high"] = pred["mean_ci_upper"]
    data["effect"] = data["y"] - data["counterfactual"]

    post_mask = data["month"] >= policy_ts
    avg_effect = data.loc[post_mask, "effect"].mean() if post_mask.any() else np.nan
    total_effect = data.loc[post_mask, "effect"].sum() if post_mask.any() else np.nan

    summary = pd.DataFrame(
        {
            "metric": [
                "avg_post_policy_treatment_effect",
                "total_post_policy_treatment_effect",
                "model_r_squared",
                "post_period_points",
            ],
            "value": [
                float(avg_effect) if pd.notna(avg_effect) else np.nan,
                float(total_effect) if pd.notna(total_effect) else np.nan,
                float(model.rsquared),
                int(post_mask.sum()),
            ],
        }
    )

    effects_path = os.path.join(config.output_dir, "causal_effects.csv")
    summary_path = os.path.join(config.output_dir, "causal_summary.csv")
    plot_path = os.path.join(config.output_dir, "causal_counterfactual.png")

    data.to_csv(effects_path, index=False)
    summary.to_csv(summary_path, index=False)

    plt.figure(figsize=(11, 5))
    plt.plot(data["month"], data["y"], label="Observed", color="#1f77b4")
    plt.plot(data["month"], data["counterfactual"], label="Counterfactual", color="#ff7f0e", linestyle="--")
    plt.fill_between(data["month"], data["cf_ci_low"], data["cf_ci_high"], color="#ff7f0e", alpha=0.2, label="95% CI")
    plt.axvline(policy_ts, color="red", linestyle="--", label="Policy Start")
    plt.title("Observed vs Counterfactual Outcome")
    plt.xlabel("Month")
    plt.ylabel("Outcome")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return data, summary, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate quasi-causal policy impact.")
    parser.add_argument("--treated-input", required=True)
    parser.add_argument("--date-col", default="DATE")
    parser.add_argument("--value-col", required=True)
    parser.add_argument("--policy-date", required=True)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--control-input")
    parser.add_argument("--control-value-col")
    args = parser.parse_args()

    cfg = CausalConfig(
        treated_input=args.treated_input,
        date_col=args.date_col,
        value_col=args.value_col,
        policy_date=args.policy_date,
        output_dir=args.output_dir,
        control_input=args.control_input,
        control_value_col=args.control_value_col,
    )
    data, summary, plot_path = run(cfg)
    print(f"Saved causal effects: {len(data)} rows")
    print(summary)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
