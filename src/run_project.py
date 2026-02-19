"""Convenience entrypoint for running project workflows."""

from __future__ import annotations

import argparse

from download_fred_series import run as run_fred_download
from policy_eda import PolicyEDAConfig, run as run_policy_eda
from sentiment_pipeline import SentimentConfig, run as run_sentiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run either sentiment or policy workflow.")
    subparsers = parser.add_subparsers(dest="workflow", required=True)

    fred = subparsers.add_parser("download-fred", help="Download a FRED CSV series.")
    fred.add_argument("--series-id", required=True)
    fred.add_argument("--output", required=True)

    sentiment = subparsers.add_parser("sentiment", help="Run Reddit sentiment workflow.")
    sentiment.add_argument("--query", required=True)
    sentiment.add_argument("--subreddit", default="all")
    sentiment.add_argument("--limit", type=int, default=200)
    sentiment.add_argument("--output", default="data/processed/reddit_sentiment.csv")

    policy = subparsers.add_parser("policy", help="Run policy impact EDA workflow.")
    policy.add_argument("--input", required=True)
    policy.add_argument("--output-dir", default="reports")
    policy.add_argument("--date-col", default="date")
    policy.add_argument("--value-col", required=True)
    policy.add_argument("--policy-date", required=True)

    nyc = subparsers.add_parser(
        "nyc-case",
        help="Run the predefined NYC case (HSTPA + FRED housing index + student-loan sentiment query).",
    )
    nyc.add_argument("--skip-download", action="store_true")
    nyc.add_argument("--skip-sentiment", action="store_true")
    nyc.add_argument("--skip-policy", action="store_true")
    nyc.add_argument("--sentiment-limit", type=int, default=300)

    la = subparsers.add_parser(
        "la-case",
        help="Run the predefined LA/USC case (Measure ULA + FRED housing index + student-loan sentiment query).",
    )
    la.add_argument("--skip-download", action="store_true")
    la.add_argument("--skip-sentiment", action="store_true")
    la.add_argument("--skip-policy", action="store_true")
    la.add_argument("--sentiment-limit", type=int, default=300)

    args = parser.parse_args()

    if args.workflow == "download-fred":
        df = run_fred_download(args.series_id, args.output)
        print(f"Downloaded FRED series: {len(df)} rows")
        return

    if args.workflow == "sentiment":
        cfg = SentimentConfig(args.query, args.subreddit, args.limit, args.output)
        df = run_sentiment(cfg)
        print(f"Sentiment workflow complete: {len(df)} rows")
        return

    if args.workflow == "nyc-case":
        fred_out = "data/raw/nyc_hpi_fred.csv"
        if not args.skip_download:
            fred_df = run_fred_download("ATNHPIUS35620Q", fred_out)
            print(f"Downloaded NYC metro HPI rows: {len(fred_df)}")

        if not args.skip_sentiment:
            sent_cfg = SentimentConfig(
                query="student loan forgiveness",
                subreddit="all",
                limit=args.sentiment_limit,
                output_path="data/processed/reddit_sentiment.csv",
            )
            sent_df = run_sentiment(sent_cfg)
            print(f"Sentiment workflow complete: {len(sent_df)} rows")

        if not args.skip_policy:
            pol_cfg = PolicyEDAConfig(
                input_path=fred_out,
                output_dir="reports",
                date_col="DATE",
                value_col="ATNHPIUS35620Q",
                policy_date="2019-06-14",
            )
            monthly, summary, plot_path = run_policy_eda(pol_cfg)
            print(f"Policy workflow complete: {len(monthly)} monthly points")
            print(summary)
            print(f"Plot: {plot_path}")
        return

    if args.workflow == "la-case":
        fred_out = "data/raw/la_hpi_fred.csv"
        if not args.skip_download:
            fred_df = run_fred_download("ATNHPIUS31080Q", fred_out)
            print(f"Downloaded LA metro HPI rows: {len(fred_df)}")

        if not args.skip_sentiment:
            sent_cfg = SentimentConfig(
                query="student loan forgiveness",
                subreddit="all",
                limit=args.sentiment_limit,
                output_path="data/processed/reddit_sentiment.csv",
            )
            sent_df = run_sentiment(sent_cfg)
            print(f"Sentiment workflow complete: {len(sent_df)} rows")

        if not args.skip_policy:
            pol_cfg = PolicyEDAConfig(
                input_path=fred_out,
                output_dir="reports",
                date_col="DATE",
                value_col="ATNHPIUS31080Q",
                policy_date="2023-04-01",
            )
            monthly, summary, plot_path = run_policy_eda(pol_cfg)
            print(f"Policy workflow complete: {len(monthly)} monthly points")
            print(summary)
            print(f"Plot: {plot_path}")
        return

    cfg = PolicyEDAConfig(args.input, args.output_dir, args.date_col, args.value_col, args.policy_date)
    monthly, summary, plot_path = run_policy_eda(cfg)
    print(f"Policy workflow complete: {len(monthly)} monthly points")
    print(summary)
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
