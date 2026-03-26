"""
Eval Case Generator — reads eval_suite.yaml, filters the test set,
and writes one JSONL file per case into eval/cases/.
"""

import json
import os
import sys

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply filter criteria from eval_suite.yaml to a DataFrame."""
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        if key.endswith("_min"):
            col = key.rsplit("_min", 1)[0]
            mask &= df[col] >= value
        elif key.endswith("_max"):
            col = key.rsplit("_max", 1)[0]
            mask &= df[col] <= value
        else:
            mask &= df[key] == value
    return df[mask]


def build_eval_cases(
    features_path: str = None,
    suite_path: str = None,
    output_dir: str = None,
):
    """Read eval_suite.yaml, filter the test set, write case files + manifest."""
    if features_path is None:
        features_path = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
    if suite_path is None:
        suite_path = os.path.join(PROJECT_ROOT, "eval", "eval_suite.yaml")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "eval", "cases")

    df = pd.read_csv(features_path)
    test = df[df["gameweek"] > 30].copy()
    print(f"Test set: {len(test)} rows (GW31+)")

    with open(suite_path) as f:
        suite = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    for category_name, category in suite["categories"].items():
        for case in category["cases"]:
            filtered = apply_filters(test, case.get("filter", {}))
            count = len(filtered)

            if count == 0:
                print(f"  WARN: {case['name']} matched 0 rows — skipping")
                continue

            case_file = os.path.join(output_dir, f"{case['name']}.jsonl")
            filtered.to_json(case_file, orient="records", lines=True)

            manifest.append({
                "name": case["name"],
                "category": category_name,
                "description": case.get("description", ""),
                "criteria": case.get("criteria", ""),
                "expected_range": case.get("expected_range", [0, 20]),
                "count": count,
                "file": case_file,
            })

            print(f"  {case['name']}: {count} rows")

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nBuilt {len(manifest)} eval cases → {output_dir}")
    return manifest


if __name__ == "__main__":
    build_eval_cases()
