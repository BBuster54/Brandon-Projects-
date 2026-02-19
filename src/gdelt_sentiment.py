"""No-key sentiment ingestion using GDELT DOC API + VADER scoring."""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


API_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass
class GDELTConfig:
    query: str
    max_records: int
    output_path: str


def _fetch_articles(query: str, max_records: int) -> list[dict]:
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(max_records),
        "sort": "DateDesc",
        "format": "json",
    }
    url = f"{API_BASE}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("articles", [])


def _label_from_compound(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def run(config: GDELTConfig) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    articles = _fetch_articles(config.query, config.max_records)

    rows: list[dict] = []
    for i, art in enumerate(articles):
        title = art.get("title") or ""
        body = art.get("seendate") or ""
        text = f"{title} {body}".strip()
        score = analyzer.polarity_scores(text)

        seen = art.get("seendate")
        try:
            created = pd.to_datetime(seen, utc=True)
        except Exception:
            created = pd.to_datetime(datetime.utcnow(), utc=True)

        rows.append(
            {
                "id": art.get("url") or f"gdelt_{i}",
                "created_utc": created,
                "title": title,
                "body": art.get("domain") or "",
                "score": 0,
                "num_comments": 0,
                "compound": score["compound"],
                "positive": score["pos"],
                "neutral": score["neu"],
                "negative": score["neg"],
                "sentiment": _label_from_compound(score["compound"]),
                "query": config.query,
                "subreddit": "gdelt_news",
                "url": art.get("url") or "",
                "source": "gdelt",
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["created_utc"]).dt.date
        df = df.sort_values("created_utc", ascending=False)

    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    df.to_csv(config.output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GDELT articles and run sentiment scoring.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-records", type=int, default=250)
    parser.add_argument("--output", default="data/processed/gdelt_sentiment.csv")
    args = parser.parse_args()

    cfg = GDELTConfig(query=args.query, max_records=args.max_records, output_path=args.output)
    df = run(cfg)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
