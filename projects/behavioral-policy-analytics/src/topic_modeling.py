"""Topic modeling over social posts with LDA and temporal topic evolution."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class TopicConfig:
    input_path: str
    output_dir: str
    n_topics: int
    top_k_terms: int


def _clean_text(df: pd.DataFrame) -> pd.Series:
    title = df.get("title", pd.Series([""] * len(df)))
    body = df.get("body", pd.Series([""] * len(df)))
    text = (title.fillna("") + " " + body.fillna("")).str.strip()
    return text


def run(config: TopicConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    os.makedirs(config.output_dir, exist_ok=True)

    df = pd.read_csv(config.input_path)
    if df.empty:
        raise ValueError("Topic modeling input is empty.")

    text = _clean_text(df)
    vectorizer = CountVectorizer(stop_words="english", max_features=2500, min_df=2)
    X = vectorizer.fit_transform(text)

    lda = LatentDirichletAllocation(n_components=config.n_topics, random_state=42)
    doc_topics = lda.fit_transform(X)

    terms = vectorizer.get_feature_names_out()
    topic_rows = []
    for i, comp in enumerate(lda.components_):
        idx = comp.argsort()[::-1][: config.top_k_terms]
        topic_rows.append(
            {
                "topic": i,
                "top_terms": ", ".join(terms[j] for j in idx),
            }
        )

    topic_keywords = pd.DataFrame(topic_rows)
    topic_keywords_path = os.path.join(config.output_dir, "topic_keywords.csv")
    topic_keywords.to_csv(topic_keywords_path, index=False)

    if "created_utc" in df.columns:
        dt = pd.to_datetime(df["created_utc"])
    else:
        dt = pd.Timestamp.today().normalize() + pd.to_timedelta(range(len(df)), unit="D")

    topic_assign = pd.DataFrame(doc_topics, columns=[f"topic_{i}" for i in range(config.n_topics)])
    topic_assign["month"] = pd.to_datetime(dt).to_period("M").to_timestamp()
    topic_evolution = topic_assign.groupby("month", as_index=False).mean()

    evolution_path = os.path.join(config.output_dir, "topic_evolution.csv")
    topic_evolution.to_csv(evolution_path, index=False)

    plot_path = os.path.join(config.output_dir, "topic_evolution.png")
    plt.figure(figsize=(11, 5))
    for i in range(config.n_topics):
        plt.plot(topic_evolution["month"], topic_evolution[f"topic_{i}"], label=f"Topic {i}")
    plt.title("Topic Evolution Over Time")
    plt.xlabel("Month")
    plt.ylabel("Average Topic Weight")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    return topic_keywords, topic_evolution, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LDA topic modeling.")
    parser.add_argument("--input", default="data/processed/sentiment.csv")
    parser.add_argument("--output-dir", default="reports/topics")
    parser.add_argument("--n-topics", type=int, default=5)
    parser.add_argument("--top-k-terms", type=int, default=10)
    args = parser.parse_args()

    cfg = TopicConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        n_topics=args.n_topics,
        top_k_terms=args.top_k_terms,
    )
    keywords, evolution, plot_path = run(cfg)
    print(keywords)
    print(f"Saved evolution rows: {len(evolution)}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
