"""
Eval Runner — single command to evaluate all models against the eval suite.

Usage:
    python eval/run_eval.py --tag v3-baseline
    python eval/run_eval.py --tag v3-baseline --compare-baseline

Loads existing prediction files (no re-inference needed), scores each model
against every eval case, and saves timestamped results.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from eval.build_cases import build_eval_cases
from eval.checks.output_checks import check_distribution, run_all_checks
from eval.compare import compare_runs


def load_predictions(eval_dir: str) -> dict:
    """Load all existing prediction files and the test features."""
    features = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "features.csv"))
    test = features[features["gameweek"] > 30].reset_index(drop=True)

    models = {}

    # XGBoost — predictions stored in xgboost_results.json with the test set
    # We need to re-predict from the model, but for eval we use the test set actuals
    # and load predictions from the LLM prediction files (which have the same index order)
    xgb_path = os.path.join(PROJECT_ROOT, "models", "xgboost_fpl.json")
    if os.path.exists(xgb_path):
        try:
            import xgboost as xgb

            model = xgb.XGBRegressor()
            model.load_model(xgb_path)
            drop_cols = ["player_id", "player_name", "target_points", "gameweek"]
            X_test = test.drop(columns=drop_cols)
            X_test = pd.get_dummies(X_test, columns=["position"])
            # Align columns with what the model expects
            expected_cols = model.get_booster().feature_names
            X_test = X_test.reindex(columns=expected_cols, fill_value=0)
            preds = model.predict(X_test)
            models["xgboost"] = {
                "predictions": [{"index": i, "actual": int(test.iloc[i]["target_points"]),
                                 "parsed_prediction": float(p), "parse_success": True}
                                for i, p in enumerate(preds)],
                "is_llm": False,
            }
        except Exception as e:
            print(f"  WARN: Could not load XGBoost model: {e}")

    # LLM strategies
    for strategy in ["zero_shot", "few_shot", "chain_of_thought", "fine_tuned"]:
        path = os.path.join(eval_dir, f"llm_{strategy}_predictions.json")
        if os.path.exists(path):
            with open(path) as f:
                preds = json.load(f)
            models[f"llm_{strategy}"] = {"predictions": preds, "is_llm": True}

    return models, test


def evaluate_case(predictions: list, case_df: pd.DataFrame, case_meta: dict, is_llm: bool) -> dict:
    """Evaluate a model's predictions against a single eval case."""
    expected_range = case_meta.get("expected_range", [0, 20])

    # Match predictions to case rows by test-set index
    case_indices = set(case_df.index.tolist())
    matched_preds = []
    matched_actuals = []
    check_results = {"total": 0, "failures": 0, "warnings": 0}

    for pred in predictions:
        idx = pred["index"]
        if idx not in case_indices:
            continue
        if not pred.get("parse_success", True):
            check_results["failures"] += 1
            check_results["total"] += 1
            continue

        parsed = pred["parsed_prediction"]
        actual = pred["actual"]
        matched_preds.append(parsed)
        matched_actuals.append(actual)

        if is_llm:
            checks = run_all_checks(str(int(parsed)), expected_range)
            check_results["total"] += 1
            if not checks["all_passed"]:
                check_results["failures"] += 1

    if len(matched_preds) == 0:
        return {"mae": None, "count": 0, "error": "No matching predictions"}

    preds_arr = np.array(matched_preds)
    actuals_arr = np.array(matched_actuals)
    errors = np.abs(actuals_arr - preds_arr)

    in_expected = np.mean([expected_range[0] <= p <= expected_range[1] for p in preds_arr])

    return {
        "mae": round(float(np.mean(errors)), 3),
        "rmse": round(float(np.sqrt(np.mean(errors**2))), 3),
        "within_1": round(float(np.mean(errors <= 1)), 3),
        "within_3": round(float(np.mean(errors <= 3)), 3),
        "in_expected_range_pct": round(float(in_expected), 3),
        "count": len(matched_preds),
        "mean_prediction": round(float(np.mean(preds_arr)), 2),
        "mean_actual": round(float(np.mean(actuals_arr)), 2),
        "checks": check_results if is_llm else None,
    }


def run_eval(tag: str, compare_baseline: bool = False):
    """Run the full eval suite and save results."""
    eval_dir = os.path.join(PROJECT_ROOT, "eval")
    cases_dir = os.path.join(eval_dir, "cases")

    # Step 1: Build/refresh eval cases
    print("=" * 60)
    print("STEP 1: Building eval cases from eval_suite.yaml")
    print("=" * 60)
    manifest = build_eval_cases()

    # Step 2: Load predictions
    print("\n" + "=" * 60)
    print("STEP 2: Loading model predictions")
    print("=" * 60)
    models, test_df = load_predictions(eval_dir)
    print(f"  Loaded {len(models)} models: {', '.join(models.keys())}")

    # Step 3: Evaluate each model against each case
    print("\n" + "=" * 60)
    print("STEP 3: Evaluating models against eval cases")
    print("=" * 60)

    results = {
        "tag": tag,
        "timestamp": datetime.now().isoformat(),
        "test_size": len(test_df),
        "models": {},
    }

    for model_name, model_data in models.items():
        predictions = model_data["predictions"]
        is_llm = model_data["is_llm"]

        # Build a lookup from index for the case matching
        # First, we need the test_df indexed properly
        model_results = {"categories": {}, "overall": {}, "checks": {}}
        all_preds = []
        all_actuals = []

        print(f"\n  --- {model_name} ---")

        for case_meta in manifest:
            case_df = pd.read_json(case_meta["file"], lines=True)
            # Re-index case_df to match the test set indices
            # The case JSONL rows correspond to test-set rows; find their indices
            case_indices = []
            for _, case_row in case_df.iterrows():
                matches = test_df[
                    (test_df["player_id"] == case_row["player_id"])
                    & (test_df["gameweek"] == case_row["gameweek"])
                ]
                if len(matches) > 0:
                    case_indices.append(matches.index[0])

            case_indexed = pd.DataFrame(index=case_indices)
            case_result = evaluate_case(predictions, case_indexed, case_meta, is_llm)

            category = case_meta["category"]
            if category not in model_results["categories"]:
                model_results["categories"][category] = {"cases": {}}

            model_results["categories"][category]["cases"][case_meta["name"]] = {
                **case_result,
                "description": case_meta["description"],
            }

            if case_result["mae"] is not None:
                print(f"    {case_meta['name']}: MAE {case_result['mae']:.3f} ({case_result['count']} rows)")

        # Category-level rollups
        for cat_name, cat_data in model_results["categories"].items():
            case_maes = [c["mae"] for c in cat_data["cases"].values() if c["mae"] is not None]
            case_counts = [c["count"] for c in cat_data["cases"].values() if c["mae"] is not None]
            if case_maes:
                # Weighted average by count
                total = sum(case_counts)
                cat_data["category_mae"] = round(
                    sum(m * c for m, c in zip(case_maes, case_counts)) / total, 3
                )
                cat_data["total_count"] = total

        # Overall metrics from all predictions
        for pred in predictions:
            if pred.get("parse_success", True) and pred["parsed_prediction"] is not None:
                all_preds.append(pred["parsed_prediction"])
                all_actuals.append(pred["actual"])

        if all_preds:
            preds_arr = np.array(all_preds)
            actuals_arr = np.array(all_actuals)
            errors = np.abs(actuals_arr - preds_arr)
            model_results["overall"] = {
                "mae": round(float(np.mean(errors)), 3),
                "rmse": round(float(np.sqrt(np.mean(errors**2))), 3),
                "within_1": round(float(np.mean(errors <= 1)), 3),
                "within_3": round(float(np.mean(errors <= 3)), 3),
                "total_predictions": len(all_preds),
            }

            # Distribution check for LLM models
            if is_llm:
                dist_check = check_distribution([int(p) for p in all_preds])
                model_results["checks"]["distribution"] = dist_check

        results["models"][model_name] = model_results

    # Step 4: Save results
    print("\n" + "=" * 60)
    print("STEP 4: Saving results")
    print("=" * 60)

    date_str = datetime.now().strftime("%Y-%m-%d")
    results_dir = os.path.join(eval_dir, "results", f"{date_str}_{tag}")
    os.makedirs(results_dir, exist_ok=True)

    scores_path = os.path.join(results_dir, "scores.json")
    with open(scores_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Scores → {scores_path}")

    # Generate summary markdown
    summary = generate_summary(results, tag)
    summary_path = os.path.join(results_dir, "summary.md")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"  Summary → {summary_path}")

    # Step 5: Compare against baseline
    if compare_baseline:
        baseline_path = os.path.join(eval_dir, "baseline", "scores.json")
        if os.path.exists(baseline_path):
            print("\n" + "=" * 60)
            print("STEP 5: Comparing against baseline")
            print("=" * 60)
            with open(baseline_path) as f:
                baseline = json.load(f)
            comparison = compare_runs(results, baseline)
            reg_path = os.path.join(results_dir, "regressions.json")
            with open(reg_path, "w") as f:
                json.dump(comparison, f, indent=2)
            print(f"  Regressions → {reg_path}")
            if comparison["has_regressions"]:
                print(f"\n  WARNING: {len(comparison['regressions'])} regressions detected!")
                for reg in comparison["regressions"]:
                    print(f"    {reg['model']} / {reg['level']}: "
                          f"MAE {reg['baseline_mae']:.3f} → {reg['new_mae']:.3f} "
                          f"(+{reg['delta']:.3f})")
            else:
                print("  No regressions detected.")
        else:
            print(f"\n  No baseline found at {baseline_path} — skipping comparison.")
            print(f"  To set baseline: cp {scores_path} {baseline_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<22} {'MAE':>6} {'RMSE':>6} {'±1':>6} {'±3':>6}")
    print("-" * 50)
    for model_name, model_data in results["models"].items():
        o = model_data["overall"]
        print(f"{model_name:<22} {o['mae']:>6.3f} {o['rmse']:>6.3f} "
              f"{o['within_1']:>5.1%} {o['within_3']:>5.1%}")

    print(f"\n  Full results: {results_dir}/")
    return results


def generate_summary(results: dict, tag: str) -> str:
    """Generate a markdown summary of the eval run."""
    lines = [
        f"# Eval Run: {tag}",
        f"**Date:** {results['timestamp'][:10]}",
        f"**Test set:** {results['test_size']} player-gameweeks",
        "",
        "## Overall",
        "",
        "| Model | MAE | RMSE | Within ±1 | Within ±3 |",
        "|:------|----:|-----:|----------:|----------:|",
    ]

    for model_name, model_data in results["models"].items():
        o = model_data["overall"]
        lines.append(
            f"| {model_name} | {o['mae']:.3f} | {o['rmse']:.3f} | "
            f"{o['within_1']:.1%} | {o['within_3']:.1%} |"
        )

    lines.extend(["", "## Category Breakdown", ""])

    # Collect all categories
    categories = set()
    for model_data in results["models"].values():
        categories.update(model_data["categories"].keys())

    for cat in sorted(categories):
        lines.append(f"### {cat}")
        lines.append("")
        lines.append("| Case | " + " | ".join(results["models"].keys()) + " | Count |")
        lines.append("|:-----|" + "|".join(["------:" for _ in results["models"]]) + "|------:|")

        # Get all cases in this category
        case_names = set()
        for model_data in results["models"].values():
            if cat in model_data["categories"]:
                case_names.update(model_data["categories"][cat]["cases"].keys())

        for case_name in sorted(case_names):
            row = [case_name]
            count = 0
            for model_data in results["models"].values():
                case_data = model_data["categories"].get(cat, {}).get("cases", {}).get(case_name, {})
                mae = case_data.get("mae")
                row.append(f"{mae:.3f}" if mae is not None else "—")
                count = max(count, case_data.get("count", 0))
            row.append(str(count))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Distribution checks
    lines.extend(["## Output Quality (LLM models)", ""])
    for model_name, model_data in results["models"].items():
        dist = model_data.get("checks", {}).get("distribution")
        if dist:
            status = "PASS" if dist["pass"] else "FAIL (mode collapse)"
            lines.append(f"- **{model_name}**: {dist['unique_values']} unique values "
                         f"({dist['values']}) — {status}")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FPL eval suite")
    parser.add_argument("--tag", required=True, help="Tag for this eval run (e.g. v3-baseline)")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Compare results against saved baseline")
    args = parser.parse_args()
    run_eval(args.tag, args.compare_baseline)
