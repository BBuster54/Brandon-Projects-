"""Download a FRED time series CSV for policy impact analysis."""

from __future__ import annotations

import argparse
import os

import pandas as pd


BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def run(series_id: str, output_path: str) -> pd.DataFrame:
    url = BASE_URL.format(series_id=series_id)
    df = pd.read_csv(url)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a FRED series CSV.")
    parser.add_argument("--series-id", required=True, help="FRED series ID, e.g., ATNHPIUS35620Q")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    df = run(args.series_id, args.output)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
