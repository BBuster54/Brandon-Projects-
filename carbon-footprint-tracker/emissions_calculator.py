"""Educational emissions model. Shares factors and fixtures with the web app."""
from __future__ import annotations
import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = json.loads((ROOT / "data/factors.json").read_text())
FACTORS = {f["id"]: f for f in MODEL["factors"]}

@dataclass
class FootprintInput:
    transport: str = "car"
    commute_miles: float = 600
    electricity_kwh: float = 650
    natural_gas_therms: float = 25
    flight_miles: float = 0
    landfill_bags: float = 8
    diet: str = "average"
    household: int = 1
    scope: str = "person"

def calculate_categories(inputs: FootprintInput) -> dict[str, float]:
    quantities = [inputs.commute_miles, inputs.electricity_kwh, inputs.natural_gas_therms, inputs.flight_miles, inputs.landfill_bags]
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 or v > 10000000 for v in quantities):
        raise ValueError("Usage must be finite and between 0 and 10,000,000.")
    if isinstance(inputs.household, bool) or not math.isfinite(inputs.household) or not 1 <= inputs.household <= 1000 or int(inputs.household) != inputs.household:
        raise ValueError("Household size must be an integer between 1 and 1,000.")
    if inputs.transport not in ("car", "bus", "train", "bike", "walk") or inputs.diet not in ("average", "meat", "vegetarian", "vegan") or inputs.scope not in ("person", "household"):
        raise ValueError("Invalid transport, diet or scope.")
    personal = inputs.household if inputs.scope == "household" else 1
    shared = 1 / inputs.household if inputs.scope == "person" else 1
    return {
        "Transportation": FACTORS[inputs.transport]["value"] * inputs.commute_miles * personal,
        "Electricity": FACTORS["electricity"]["value"] * inputs.electricity_kwh * shared,
        "Home heating": FACTORS["gas"]["value"] * inputs.natural_gas_therms * shared,
        "Air travel": FACTORS["flights"]["value"] * inputs.flight_miles * personal,
        "Waste": FACTORS["waste"]["value"] * inputs.landfill_bags * shared,
        "Diet": FACTORS[inputs.diet]["value"] * 30 * personal,
    }

def from_web(inputs: dict) -> FootprintInput:
    distance = float(inputs["distance"]) / (1.609344 if inputs["distanceUnit"] == "km" else 1)
    if inputs["period"] == "week":
        distance *= 52 / 12
    return FootprintInput(inputs["transport"], distance, float(inputs["electricity"]), float(inputs["gas"]),
        float(inputs["flights"]) / (1.609344 if inputs["distanceUnit"] == "km" else 1),
        float(inputs["waste"]), inputs["diet"], int(inputs["household"]), inputs["scope"])

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["car","bus","train","bike","walk"], default="car")
    commute = parser.add_mutually_exclusive_group()
    commute.add_argument("--commute-miles", type=float, help="Monthly per-person commute miles; defaults to 600.")
    commute.add_argument("--commute-miles-per-week", type=float, help="Weekly miles converted with 52/12.")
    parser.add_argument("--electricity-kwh", type=float, default=650)
    parser.add_argument("--natural-gas-therms", type=float, default=25)
    parser.add_argument("--flight-miles", type=float, default=0)
    parser.add_argument("--landfill-bags", type=float, default=8)
    parser.add_argument("--diet", choices=["average","meat","vegetarian","vegan"], default="average")
    parser.add_argument("--household", type=int, default=1)
    parser.add_argument("--scope", choices=["person","household"], default="person")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--no-chart", action="store_true", help="Run without Matplotlib.")
    return parser.parse_args(argv)

def input_from_args(args):
    miles = args.commute_miles if args.commute_miles is not None else 600
    if args.commute_miles_per_week is not None:
        miles = args.commute_miles_per_week * 52 / 12
    return FootprintInput(args.transport, miles, args.electricity_kwh, args.natural_gas_therms,
        args.flight_miles, args.landfill_bags, args.diet, args.household, args.scope)

def main():
    args = parse_arguments()
    try:
        inputs = input_from_args(args)
        categories = calculate_categories(inputs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "monthly_footprint.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["category", "kg_co2e", "factor_version"])
        writer.writerows([k, v, MODEL["version"]] for k, v in categories.items())
    for label, value in categories.items():
        print(f"{label:18} {value:,.1f} kg CO₂e")
    total = sum(categories.values())
    factor_ids = [inputs.transport, "electricity", "gas", "flights", "waste", inputs.diet]
    widths = [value * MODEL["uncertainty"][FACTORS[key]["confidence"]] for value, key in zip(categories.values(), factor_ids)]
    print(f"Total: {total:,.1f} kg CO₂e/month ({inputs.scope})")
    print(f"Assumed range: {total-sum(widths):,.1f}–{total+sum(widths):,.1f}; summed bounds, not a statistical interval.")
    if not args.no_chart:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.barh(list(categories), list(categories.values()), color=["#a84328","#3157c5","#8252ad","#196c83","#8b6514","#32735e"])
        axis.set_xlabel("kg CO₂e / month")
        axis.set_title("Monthly footprint — model " + MODEL["version"])
        figure.tight_layout()
        figure.savefig(args.output_dir / "monthly_footprint.png", dpi=180)
        plt.close(figure)

if __name__ == "__main__":
    main()

