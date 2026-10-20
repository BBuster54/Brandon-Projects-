import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emissions_calculator import calculate_categories, from_web, FootprintInput, parse_arguments, input_from_args

FIXTURES = json.loads((Path(__file__).parent / "fixtures.json").read_text())

@pytest.mark.parametrize("fixture", FIXTURES, ids=[f["name"] for f in FIXTURES])
def test_shared_fixture(fixture):
    result = calculate_categories(from_web(fixture["input"]))
    assert result["Transportation"] == pytest.approx(fixture["transport"])
    assert sum(result.values()) == pytest.approx(fixture["total"])

def test_weekly_cli_option():
    args = parse_arguments(["--commute-miles-per-week", "150"])
    assert input_from_args(args).commute_miles == pytest.approx(650)

def test_cli_commute_options_are_exclusive():
    with pytest.raises(SystemExit):
        parse_arguments(["--commute-miles", "600", "--commute-miles-per-week", "150"])

@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), 10000001])
def test_invalid_usage(value):
    with pytest.raises(ValueError):
        calculate_categories(FootprintInput(electricity_kwh=value))

def test_defaults_match_web():
    assert sum(calculate_categories(FootprintInput()).values()) == pytest.approx(862.05)

