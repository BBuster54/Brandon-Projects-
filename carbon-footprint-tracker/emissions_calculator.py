"""Estimate a monthly household carbon footprint and create a category chart.

The factors below are illustrative estimates intended for an educational portfolio
project. They are not a substitute for a verified carbon accounting method.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt


TRANSPORT_FACTORS = {"car": 0.404, "bus": 0.14, "train": 0.09, "bike": 0.0, "walk": 0.0}
ELECTRICITY_FACTOR = 0.371  # kg CO₂e per kWh; illustrative U.S. average
NATURAL_GAS_FACTOR = 5.3  # kg CO₂e per therm
FLIGHT_FACTOR = 0.255  # kg CO₂e per passenger mile
WASTE_FACTOR = 12.0  # kg CO₂e per bag of landfill waste


@dataclass
class FootprintInput:
    transport: str
    commute_miles: float
    electricity_kwh: float
    natural_gas_therms: float
    flight_miles: float
    landfill_bags: float


def calculate_categories(inputs: FootprintInput) -> dict[str, float]:
    """Return estimated monthly emissions in kilograms of CO₂e by category."""
    return {
        "Transportation": TRANSPORT_FACTORS[inputs.transport] * inputs.commute_miles,
        "Electricity": ELECTRICITY_FACTOR * inputs.electricity_kwh,
        "Home heating": NATURAL_GAS_FACTOR * inputs.natural_gas_therms,
        "Air travel": FLIGHT_FACTOR * inputs.flight_miles,
        "Waste": WASTE_FACTOR * inputs.landfill_bags,
    }


def save_chart(categories: dict[str, float], output_path: Path) -> None:
    labels, values = zip(*categories.items())
    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(labels, values, color=["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6", "#ef4444"])
    axis.set_title("Estimated Monthly Carbon Footprint")
    axis.set_ylabel("Kilograms of CO₂e")
    axis.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,.1f}", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_summary(inputs: FootprintInput, categories: dict[str, float], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["category", "kg_co2e"])
        writer.writeheader()
        writer.writerows({"category": category, "kg_co2e": round(value, 2)} for category, value in categories.items())


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate a monthly carbon footprint and save a chart.")
    parser.add_argument("--transport", choices=TRANSPORT_FACTORS, default="car")
    parser.add_argument("--commute-miles", type=float, default=600, help="Monthly miles travelled by the selected mode.")
    parser.add_argument("--electricity-kwh", type=float, default=650, help="Monthly household electricity use.")
    parser.add_argument("--natural-gas-therms", type=float, default=25, help="Monthly household natural-gas use.")
    parser.add_argument("--flight-miles", type=float, default=0, help="Monthly passenger flight miles.")
    parser.add_argument("--landfill-bags", type=float, default=8, help="Monthly bags of landfill waste.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    values = [args.commute_miles, args.electricity_kwh, args.natural_gas_therms, args.flight_miles, args.landfill_bags]
    if any(value < 0 for value in values):
        raise SystemExit("All usage values must be zero or greater.")

    inputs = FootprintInput(
        transport=args.transport,
        commute_miles=args.commute_miles,
        electricity_kwh=args.electricity_kwh,
        natural_gas_therms=args.natural_gas_therms,
        flight_miles=args.flight_miles,
        landfill_bags=args.landfill_bags,
    )
    categories = calculate_categories(inputs)
    total = sum(categories.values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = args.output_dir / "monthly_footprint.png"
    csv_path = args.output_dir / "monthly_footprint.csv"
    save_chart(categories, chart_path)
    save_summary(inputs, categories, csv_path)

    print("Estimated monthly carbon footprint")
    print("-" * 36)
    for category, value in categories.items():
        print(f"{category:18} {value:8.1f} kg CO₂e")
    print("-" * 36)
    print(f"Total              {total:8.1f} kg CO₂e")
    print(f"Chart saved to {chart_path}")
    print(f"CSV summary saved to {csv_path}")


if __name__ == "__main__":
    main()
