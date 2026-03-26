"""
Regression Detector — compares two eval runs and flags performance changes.

Usage:
    python eval/compare.py --new eval/results/2026-03-26_v3/scores.json \
                           --baseline eval/baseline/scores.json
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def compare_runs(
    new_results: dict,
    baseline_results: dict,
    threshold: float = 0.1,
) -> dict:
    """
    Compare two eval runs. Flag regressions where MAE increased
    by more than threshold, and improvements where it decreased.
    """
    regressions = []
    improvements = []

    for model_name in new_results["models"]:
        if model_name not in baseline_results["models"]:
            continue

        new_model = new_results["models"][model_name]
        base_model = baseline_results["models"][model_name]

        # Overall regression check
        new_mae = new_model["overall"]["mae"]
        base_mae = base_model["overall"]["mae"]
        delta = round(new_mae - base_mae, 3)

        entry = {
            "model": model_name,
            "level": "overall",
            "baseline_mae": base_mae,
            "new_mae": new_mae,
            "delta": delta,
        }
        if new_mae > base_mae + threshold:
            regressions.append(entry)
        elif new_mae < base_mae - threshold:
            improvements.append(entry)

        # Per-category regression check
        for cat_name, cat_data in new_model.get("categories", {}).items():
            base_cat = base_model.get("categories", {}).get(cat_name, {})
            if not base_cat or "category_mae" not in cat_data or "category_mae" not in base_cat:
                continue

            new_cat_mae = cat_data["category_mae"]
            base_cat_mae = base_cat["category_mae"]
            delta = round(new_cat_mae - base_cat_mae, 3)

            entry = {
                "model": model_name,
                "level": f"category:{cat_name}",
                "baseline_mae": base_cat_mae,
                "new_mae": new_cat_mae,
                "delta": delta,
            }
            if new_cat_mae > base_cat_mae + threshold:
                regressions.append(entry)
            elif new_cat_mae < base_cat_mae - threshold:
                improvements.append(entry)

        # Per-case regression check
        for cat_name, cat_data in new_model.get("categories", {}).items():
            base_cases = base_model.get("categories", {}).get(cat_name, {}).get("cases", {})
            for case_name, case_data in cat_data.get("cases", {}).items():
                base_case = base_cases.get(case_name, {})
                if not base_case or case_data.get("mae") is None or base_case.get("mae") is None:
                    continue

                new_case_mae = case_data["mae"]
                base_case_mae = base_case["mae"]
                delta = round(new_case_mae - base_case_mae, 3)

                entry = {
                    "model": model_name,
                    "level": f"case:{case_name}",
                    "baseline_mae": base_case_mae,
                    "new_mae": new_case_mae,
                    "delta": delta,
                }
                if new_case_mae > base_case_mae + threshold:
                    regressions.append(entry)
                elif new_case_mae < base_case_mae - threshold:
                    improvements.append(entry)

    return {
        "regressions": regressions,
        "improvements": improvements,
        "has_regressions": len(regressions) > 0,
        "summary": (
            f"{len(regressions)} regression(s), {len(improvements)} improvement(s) "
            f"(threshold: ±{threshold})"
        ),
    }


def print_comparison(comparison: dict):
    """Pretty-print a comparison report."""
    if comparison["has_regressions"]:
        print(f"\nWARNING: {len(comparison['regressions'])} REGRESSIONS DETECTED")
        print("-" * 50)
        for r in comparison["regressions"]:
            print(f"  {r['model']} / {r['level']}")
            print(f"    MAE: {r['baseline_mae']:.3f} → {r['new_mae']:.3f} (+{r['delta']:.3f})")
    else:
        print("\nNo regressions detected.")

    if comparison["improvements"]:
        print(f"\n{len(comparison['improvements'])} IMPROVEMENTS:")
        print("-" * 50)
        for i in comparison["improvements"]:
            print(f"  {i['model']} / {i['level']}")
            print(f"    MAE: {i['baseline_mae']:.3f} → {i['new_mae']:.3f} ({i['delta']:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two eval runs")
    parser.add_argument("--new", required=True, help="Path to new scores.json")
    parser.add_argument("--baseline", required=True, help="Path to baseline scores.json")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="MAE delta threshold for flagging (default: 0.1)")
    args = parser.parse_args()

    with open(args.new) as f:
        new = json.load(f)
    with open(args.baseline) as f:
        baseline = json.load(f)

    comparison = compare_runs(new, baseline, args.threshold)
    print_comparison(comparison)

    out_path = os.path.splitext(args.new)[0] + "_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved to {out_path}")
