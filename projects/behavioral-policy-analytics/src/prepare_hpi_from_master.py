"""Prepare LA/NYC HPI series from the FHFA master HPI file.

Converts quarterly rows into DATE + index CSVs that the existing pipeline expects.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def _to_quarter_start(year: pd.Series, quarter: pd.Series) -> pd.Series:
    p = pd.PeriodIndex(year=year.astype(int), quarter=quarter.astype(int), freq="Q")
    return p.to_timestamp(how="start")


def _extract(df: pd.DataFrame, place_id: str, out_col: str) -> pd.DataFrame:
    part = df[(df["frequency"] == "quarterly") & (df["place_id"].astype(str) == str(place_id))].copy()
    if part.empty:
        raise ValueError(f"No rows found for place_id={place_id}")

    part["DATE"] = _to_quarter_start(part["yr"], part["period"])
    part = part.sort_values("DATE")

    # Prefer seasonally adjusted index if available.
    value = pd.to_numeric(part["index_sa"], errors="coerce")
    value = value.fillna(pd.to_numeric(part["index_nsa"], errors="coerce"))

    out = pd.DataFrame({"DATE": part["DATE"].dt.strftime("%Y-%m-%d"), out_col: value})
    out = out.dropna().drop_duplicates(subset=["DATE"]).reset_index(drop=True)
    return out


def run(master_path: str, la_out: str, nyc_out: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["frequency", "place_id", "yr", "period", "index_nsa", "index_sa"]
    df = pd.read_csv(master_path, usecols=cols)

    # Mappings based on provided master file content.
    # LA: Los Angeles-Long Beach-Glendale, CA (MSAD) -> place_id 31084
    # NYC: New York-Jersey City-White Plains, NY-NJ (MSAD) -> place_id 35614
    la = _extract(df, place_id="31084", out_col="ATNHPIUS31080Q")
    nyc = _extract(df, place_id="35614", out_col="ATNHPIUS35620Q")

    os.makedirs(os.path.dirname(la_out), exist_ok=True)
    os.makedirs(os.path.dirname(nyc_out), exist_ok=True)
    la.to_csv(la_out, index=False)
    nyc.to_csv(nyc_out, index=False)
    return la, nyc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LA/NYC HPI files from master HPI CSV.")
    parser.add_argument("--master", required=True, help="Path to hpi_master.csv")
    parser.add_argument("--la-out", default="data/raw/la_hpi_fred.csv")
    parser.add_argument("--nyc-out", default="data/raw/nyc_hpi_fred.csv")
    args = parser.parse_args()

    la, nyc = run(args.master, args.la_out, args.nyc_out)
    print(f"Saved LA rows: {len(la)} -> {args.la_out}")
    print(f"Saved NYC rows: {len(nyc)} -> {args.nyc_out}")


if __name__ == "__main__":
    main()
