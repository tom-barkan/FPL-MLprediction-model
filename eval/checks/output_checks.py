"""
Output quality checks for LLM predictions.
Validates that model outputs are parseable, plausible, and within expected ranges.
"""

import re


def check_valid_integer(response: str) -> dict:
    """Did the model return a parseable integer?"""
    cleaned = response.strip()
    try:
        value = int(cleaned)
        return {"pass": True, "value": value, "check": "valid_integer"}
    except ValueError:
        match = re.search(r"\b(\d+)\b", cleaned)
        if match:
            return {
                "pass": True,
                "value": int(match.group(1)),
                "check": "valid_integer",
                "warning": f"Extracted from noisy output: '{cleaned[:50]}'",
            }
        return {
            "pass": False,
            "value": None,
            "check": "valid_integer",
            "error": f"No integer found in: '{cleaned[:80]}'",
        }


def check_plausible_range(value: int) -> dict:
    """Is the prediction within a plausible FPL points range? (-4 to 25)"""
    if value is None:
        return {"pass": False, "check": "plausible_range"}
    in_range = -4 <= value <= 25
    return {
        "pass": in_range,
        "value": value,
        "check": "plausible_range",
        "error": None if in_range else f"Prediction {value} outside plausible range [-4, 25]",
    }


def check_expected_range(value: int, expected_min: int, expected_max: int) -> dict:
    """Is the prediction within the expected range for this eval case?"""
    if value is None:
        return {"pass": False, "check": "expected_range"}
    in_range = expected_min <= value <= expected_max
    return {
        "pass": in_range,
        "value": value,
        "check": "expected_range",
        "detail": f"Expected [{expected_min}, {expected_max}], got {value}",
    }


def check_distribution(predictions: list) -> dict:
    """Check that predictions aren't collapsed to a single value (mode collapse)."""
    unique = set(predictions)
    collapsed = len(unique) <= 1
    return {
        "pass": not collapsed,
        "check": "distribution",
        "unique_values": len(unique),
        "values": sorted(unique),
        "error": "Mode collapse detected — all predictions identical" if collapsed else None,
    }


def run_all_checks(response: str, expected_range: list = None) -> dict:
    """Run all output checks on a single prediction and return a summary."""
    results = {}

    int_check = check_valid_integer(response)
    results["valid_integer"] = int_check

    if int_check["pass"]:
        results["plausible_range"] = check_plausible_range(int_check["value"])
        if expected_range:
            results["expected_range"] = check_expected_range(
                int_check["value"], expected_range[0], expected_range[1]
            )

    results["all_passed"] = all(r.get("pass", False) for r in results.values())
    results["parsed_value"] = int_check.get("value")

    return results
