"""Sentiment pipeline for social discussion analysis.

Collects posts from Reddit and labels sentiment with VADER.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

import pandas as pd
import praw
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class SentimentConfig:
    query: str
    subreddit: str
    limit: int
    output_path: str


class RedditSentimentCollector:
    def __init__(self) -> None:
        load_dotenv()
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "sentiment-research-app/0.1"),
        )
        self.analyzer = SentimentIntensityAnalyzer()

    def fetch_posts(self, subreddit: str, query: str, limit: int) -> pd.DataFrame:
        rows: List[dict] = []
        for submission in self.reddit.subreddit(subreddit).search(query, sort="new", limit=limit):
            text = f"{submission.title} {submission.selftext}".strip()
            scores = self.analyzer.polarity_scores(text)
            label = self._label_from_compound(scores["compound"])
            rows.append(
                {
                    "id": submission.id,
                    "created_utc": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                    "title": submission.title,
                    "body": submission.selftext,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "compound": scores["compound"],
                    "positive": scores["pos"],
                    "neutral": scores["neu"],
                    "negative": scores["neg"],
                    "sentiment": label,
                    "query": query,
                    "subreddit": subreddit,
                    "url": submission.url,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["created_utc"]).dt.date
        return df.sort_values("created_utc", ascending=False)

    @staticmethod
    def _label_from_compound(compound: float) -> str:
        if compound >= 0.05:
            return "positive"
        if compound <= -0.05:
            return "negative"
        return "neutral"


def run(config: SentimentConfig) -> pd.DataFrame:
    collector = RedditSentimentCollector()
    df = collector.fetch_posts(
        subreddit=config.subreddit,
        query=config.query,
        limit=config.limit,
    )

    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    df.to_csv(config.output_path, index=False)
    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Reddit sentiment collection and scoring.")
    parser.add_argument("--query", required=True, help="Search query, e.g., 'student loan forgiveness'.")
    parser.add_argument("--subreddit", default="all", help="Subreddit to search, default: all.")
    parser.add_argument("--limit", type=int, default=200, help="Number of posts to fetch.")
    parser.add_argument(
        "--output",
        default="data/processed/reddit_sentiment.csv",
        help="Output CSV path.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = SentimentConfig(
        query=args.query,
        subreddit=args.subreddit,
        limit=args.limit,
        output_path=args.output,
    )
    df = run(config)
    print(f"Saved {len(df)} rows to {config.output_path}")


if __name__ == "__main__":
    main()
