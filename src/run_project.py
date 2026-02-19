"""Convenience entrypoint for running all project workflows."""

from __future__ import annotations

import argparse

from analyze_sentiment import run as run_sentiment_analysis
from causal_impact import CausalConfig, run as run_causal
from cross_city_compare import CompareConfig, run as run_compare
from download_fred_series import run as run_fred_download
from gdelt_sentiment import GDELTConfig, run as run_gdelt_sentiment
from lagged_prediction import LagPredictionConfig, run as run_lag_prediction
from policy_eda import PolicyEDAConfig, run as run_policy_eda
from sentiment_pipeline import SentimentConfig, run as run_reddit_sentiment
from topic_modeling import TopicConfig, run as run_topics


def _run_sentiment_source(source: str, query: str, limit: int, output_path: str) -> int:
    if source == "gdelt":
        cfg = GDELTConfig(query=query, max_records=limit, output_path=output_path)
        df = run_gdelt_sentiment(cfg)
        return len(df)

    cfg = SentimentConfig(query=query, subreddit="all", limit=limit, output_path=output_path)
    df = run_reddit_sentiment(cfg)
    return len(df)


def _run_case(
    city_key: str,
    fred_series: str,
    value_col: str,
    policy_date: str,
    sentiment_query: str,
    sentiment_limit: int,
    sentiment_source: str,
    skip_download: bool,
    skip_sentiment: bool,
    skip_policy: bool,
) -> None:
    raw_path = f"data/raw/{city_key}_hpi_fred.csv"
    processed_sent = f"data/processed/{city_key}_sentiment.csv"
    report_dir = f"reports/{city_key}"

    if not skip_download:
        fred_df = run_fred_download(fred_series, raw_path)
        print(f"Downloaded {city_key.upper()} series rows: {len(fred_df)}")

    if not skip_sentiment:
        sent_rows = _run_sentiment_source(sentiment_source, sentiment_query, sentiment_limit, processed_sent)
        print(f"Sentiment rows ({city_key}, {sentiment_source}): {sent_rows}")

        daily, _ = run_sentiment_analysis(processed_sent, report_dir)
        print(f"Daily sentiment rows ({city_key}): {len(daily)}")

        topics_cfg = TopicConfig(
            input_path=processed_sent,
            output_dir=f"{report_dir}/topics",
            n_topics=5,
            top_k_terms=10,
        )
        _, evolution, _ = run_topics(topics_cfg)
        print(f"Topic evolution rows ({city_key}): {len(evolution)}")

    if not skip_policy:
        pol_cfg = PolicyEDAConfig(
            input_path=raw_path,
            output_dir=report_dir,
            date_col="DATE",
            value_col=value_col,
            policy_date=policy_date,
        )
        monthly, summary, _ = run_policy_eda(pol_cfg)
        print(f"Policy rows ({city_key}): {len(monthly)}")
        print(summary)

        caus_cfg = CausalConfig(
            treated_input=raw_path,
            date_col="DATE",
            value_col=value_col,
            policy_date=policy_date,
            output_dir=report_dir,
        )
        _, csum, _ = run_causal(caus_cfg)
        print(csum)

        if not skip_sentiment:
            lag_cfg = LagPredictionConfig(
                monthly_series_input=f"{report_dir}/monthly_series.csv",
                sentiment_daily_input=f"{report_dir}/sentiment_daily.csv",
                output_dir=report_dir,
                max_lag=6,
            )
            lag_df, _, _ = run_lag_prediction(lag_cfg)
            print(f"Lag model rows ({city_key}): {len(lag_df)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run project workflows.")
    subparsers = parser.add_subparsers(dest="workflow", required=True)

    fred = subparsers.add_parser("download-fred", help="Download a FRED CSV series.")
    fred.add_argument("--series-id", required=True)
    fred.add_argument("--output", required=True)

    sentiment = subparsers.add_parser("sentiment", help="Run sentiment workflow.")
    sentiment.add_argument("--query", required=True)
    sentiment.add_argument("--limit", type=int, default=250)
    sentiment.add_argument("--source", choices=["gdelt", "reddit"], default="gdelt")
    sentiment.add_argument("--output", default="data/processed/sentiment.csv")

    policy = subparsers.add_parser("policy", help="Run policy impact EDA workflow.")
    policy.add_argument("--input", required=True)
    policy.add_argument("--output-dir", default="reports")
    policy.add_argument("--date-col", default="date")
    policy.add_argument("--value-col", required=True)
    policy.add_argument("--policy-date", required=True)

    causal = subparsers.add_parser("causal", help="Run quasi-causal counterfactual analysis.")
    causal.add_argument("--treated-input", required=True)
    causal.add_argument("--date-col", default="DATE")
    causal.add_argument("--value-col", required=True)
    causal.add_argument("--policy-date", required=True)
    causal.add_argument("--output-dir", default="reports")
    causal.add_argument("--control-input")
    causal.add_argument("--control-value-col")

    predict = subparsers.add_parser("predict-lags", help="Run lagged sentiment prediction analysis.")
    predict.add_argument("--monthly-series-input", required=True)
    predict.add_argument("--sentiment-daily-input", required=True)
    predict.add_argument("--output-dir", default="reports/prediction")
    predict.add_argument("--max-lag", type=int, default=6)

    topics = subparsers.add_parser("topics", help="Run LDA topic modeling.")
    topics.add_argument("--input", default="data/processed/sentiment.csv")
    topics.add_argument("--output-dir", default="reports/topics")
    topics.add_argument("--n-topics", type=int, default=5)
    topics.add_argument("--top-k-terms", type=int, default=10)

    compare = subparsers.add_parser("compare-cities", help="Compare LA and NYC policy outcomes.")
    compare.add_argument("--la-policy-summary", default="reports/la/policy_summary.csv")
    compare.add_argument("--nyc-policy-summary", default="reports/nyc/policy_summary.csv")
    compare.add_argument("--la-monthly-series", default="reports/la/monthly_series.csv")
    compare.add_argument("--nyc-monthly-series", default="reports/nyc/monthly_series.csv")
    compare.add_argument("--la-sentiment-file", default="data/processed/la_sentiment.csv")
    compare.add_argument("--nyc-sentiment-file", default="data/processed/nyc_sentiment.csv")
    compare.add_argument("--output-dir", default="reports/comparison")

    nyc = subparsers.add_parser("nyc-case", help="Run the predefined NYC case.")
    nyc.add_argument("--skip-download", action="store_true")
    nyc.add_argument("--skip-sentiment", action="store_true")
    nyc.add_argument("--skip-policy", action="store_true")
    nyc.add_argument("--sentiment-limit", type=int, default=250)
    nyc.add_argument("--sentiment-source", choices=["gdelt", "reddit"], default="gdelt")

    la = subparsers.add_parser("la-case", help="Run the predefined LA/USC case.")
    la.add_argument("--skip-download", action="store_true")
    la.add_argument("--skip-sentiment", action="store_true")
    la.add_argument("--skip-policy", action="store_true")
    la.add_argument("--sentiment-limit", type=int, default=250)
    la.add_argument("--sentiment-source", choices=["gdelt", "reddit"], default="gdelt")

    platform = subparsers.add_parser("full-platform", help="Run LA + NYC and cross-city comparison.")
    platform.add_argument("--sentiment-limit", type=int, default=250)
    platform.add_argument("--sentiment-source", choices=["gdelt", "reddit"], default="gdelt")

    args = parser.parse_args()

    if args.workflow == "download-fred":
        df = run_fred_download(args.series_id, args.output)
        print(f"Downloaded FRED series: {len(df)} rows")
        return

    if args.workflow == "sentiment":
        rows = _run_sentiment_source(args.source, args.query, args.limit, args.output)
        print(f"Sentiment workflow complete: {rows} rows")
        return

    if args.workflow == "policy":
        cfg = PolicyEDAConfig(args.input, args.output_dir, args.date_col, args.value_col, args.policy_date)
        monthly, summary, plot_path = run_policy_eda(cfg)
        print(f"Policy workflow complete: {len(monthly)} monthly points")
        print(summary)
        print(f"Plot: {plot_path}")
        return

    if args.workflow == "causal":
        cfg = CausalConfig(
            treated_input=args.treated_input,
            date_col=args.date_col,
            value_col=args.value_col,
            policy_date=args.policy_date,
            output_dir=args.output_dir,
            control_input=args.control_input,
            control_value_col=args.control_value_col,
        )
        data, summary, plot_path = run_causal(cfg)
        print(f"Causal workflow complete: {len(data)} rows")
        print(summary)
        print(f"Plot: {plot_path}")
        return

    if args.workflow == "predict-lags":
        cfg = LagPredictionConfig(
            monthly_series_input=args.monthly_series_input,
            sentiment_daily_input=args.sentiment_daily_input,
            output_dir=args.output_dir,
            max_lag=args.max_lag,
        )
        lag_df, granger_df, plot_path = run_lag_prediction(cfg)
        print(lag_df)
        print(granger_df)
        print(f"Plot: {plot_path}")
        return

    if args.workflow == "topics":
        cfg = TopicConfig(
            input_path=args.input,
            output_dir=args.output_dir,
            n_topics=args.n_topics,
            top_k_terms=args.top_k_terms,
        )
        keywords, evolution, plot_path = run_topics(cfg)
        print(keywords)
        print(f"Evolution rows: {len(evolution)}")
        print(f"Plot: {plot_path}")
        return

    if args.workflow == "compare-cities":
        cfg = CompareConfig(
            la_policy_summary=args.la_policy_summary,
            nyc_policy_summary=args.nyc_policy_summary,
            la_monthly_series=args.la_monthly_series,
            nyc_monthly_series=args.nyc_monthly_series,
            la_sentiment_file=args.la_sentiment_file,
            nyc_sentiment_file=args.nyc_sentiment_file,
            output_dir=args.output_dir,
        )
        comp, plot_path = run_compare(cfg)
        print(comp)
        print(f"Plot: {plot_path}")
        return

    if args.workflow == "la-case":
        _run_case(
            city_key="la",
            fred_series="ATNHPIUS31080Q",
            value_col="ATNHPIUS31080Q",
            policy_date="2023-04-01",
            sentiment_query="(Measure ULA OR Los Angeles housing tax OR LA housing affordability)",
            sentiment_limit=args.sentiment_limit,
            sentiment_source=args.sentiment_source,
            skip_download=args.skip_download,
            skip_sentiment=args.skip_sentiment,
            skip_policy=args.skip_policy,
        )
        return

    if args.workflow == "nyc-case":
        _run_case(
            city_key="nyc",
            fred_series="ATNHPIUS35620Q",
            value_col="ATNHPIUS35620Q",
            policy_date="2019-06-14",
            sentiment_query="(HSTPA OR New York rent reform OR NYC rent stabilization)",
            sentiment_limit=args.sentiment_limit,
            sentiment_source=args.sentiment_source,
            skip_download=args.skip_download,
            skip_sentiment=args.skip_sentiment,
            skip_policy=args.skip_policy,
        )
        return

    if args.workflow == "full-platform":
        _run_case(
            city_key="la",
            fred_series="ATNHPIUS31080Q",
            value_col="ATNHPIUS31080Q",
            policy_date="2023-04-01",
            sentiment_query="(Measure ULA OR Los Angeles housing tax OR LA housing affordability)",
            sentiment_limit=args.sentiment_limit,
            sentiment_source=args.sentiment_source,
            skip_download=False,
            skip_sentiment=False,
            skip_policy=False,
        )
        _run_case(
            city_key="nyc",
            fred_series="ATNHPIUS35620Q",
            value_col="ATNHPIUS35620Q",
            policy_date="2019-06-14",
            sentiment_query="(HSTPA OR New York rent reform OR NYC rent stabilization)",
            sentiment_limit=args.sentiment_limit,
            sentiment_source=args.sentiment_source,
            skip_download=False,
            skip_sentiment=False,
            skip_policy=False,
        )

        cfg = CompareConfig(
            la_policy_summary="reports/la/policy_summary.csv",
            nyc_policy_summary="reports/nyc/policy_summary.csv",
            la_monthly_series="reports/la/monthly_series.csv",
            nyc_monthly_series="reports/nyc/monthly_series.csv",
            la_sentiment_file="data/processed/la_sentiment.csv",
            nyc_sentiment_file="data/processed/nyc_sentiment.csv",
            output_dir="reports/comparison",
        )
        comp, plot_path = run_compare(cfg)
        print(comp)
        print(f"Cross-city plot: {plot_path}")


if __name__ == "__main__":
    main()
