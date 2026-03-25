"""
Train an XGBoost regressor to predict FPL points per gameweek.

Reads data/processed/features.csv, trains on GW 4-30, evaluates on GW 31+.
Saves the model, evaluation results, and feature importance chart.
"""

import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(ROOT, "data", "processed", "features.csv")
MODEL_PATH = os.path.join(ROOT, "models", "xgboost_fpl.json")
EVAL_DIR = os.path.join(ROOT, "eval")

DROP_COLS = ["player_id", "player_name", "target_points", "gameweek"]


def load_and_split():
    df = pd.read_csv(FEATURES_PATH)
    train = df[df["gameweek"] <= 30].copy()
    test = df[df["gameweek"] > 30].copy()
    return train, test


def prepare_features(train, test):
    X_train = train.drop(columns=DROP_COLS)
    y_train = train["target_points"]
    X_test = test.drop(columns=DROP_COLS)
    y_test = test["target_points"]

    # One-hot encode position
    X_train = pd.get_dummies(X_train, columns=["position"])
    X_test = pd.get_dummies(X_test, columns=["position"])

    # Align columns (in case test is missing a position)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
        eval_metric="mae",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50,
    )
    return model


def evaluate(model, X_test, y_test, test_df):
    preds = model.predict(X_test)
    errors = np.abs(y_test.values - preds)

    results = {
        "mae": round(mean_absolute_error(y_test, preds), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "within_1": round(float(np.mean(errors <= 1)), 4),
        "within_3": round(float(np.mean(errors <= 3)), 4),
        "test_size": len(y_test),
        "train_size": None,  # filled by caller
    }

    # Per-position breakdown
    pos_results = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        mask = test_df["position"] == pos
        if mask.sum() == 0:
            continue
        pos_preds = preds[mask.values]
        pos_actual = y_test.values[mask.values]
        pos_results[pos] = {
            "mae": round(float(mean_absolute_error(pos_actual, pos_preds)), 4),
            "count": int(mask.sum()),
        }
    results["by_position"] = pos_results

    return results, preds


def plot_feature_importance(model, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=15, ax=ax, importance_type="gain")
    ax.set_title("XGBoost Feature Importance (Gain)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Feature importance chart saved to {output_path}")


def plot_predictions_vs_actual(y_test, preds, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax = axes[0]
    ax.scatter(y_test, preds, alpha=0.4, s=15, color="#60efff", edgecolors="none")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            "r--", linewidth=1, alpha=0.7, label="Perfect prediction")
    ax.set_xlabel("Actual Points")
    ax.set_ylabel("Predicted Points")
    ax.set_title("Predicted vs Actual")
    ax.legend()

    # Error distribution
    ax = axes[1]
    errors = preds - y_test.values
    ax.hist(errors, bins=30, color="#60efff", edgecolor="#0a1628", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Prediction Error (predicted - actual)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution (MAE: {np.mean(np.abs(errors)):.2f})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Prediction analysis chart saved to {output_path}")


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    print("Loading data...")
    train, test = load_and_split()
    print(f"Train: {len(train)} rows (GW 4-30)")
    print(f"Test:  {len(test)} rows (GW 31+)")

    X_train, y_train, X_test, y_test = prepare_features(train, test)
    print(f"Features: {list(X_train.columns)}\n")

    print("Training XGBoost...")
    model = train_model(X_train, y_train, X_test, y_test)

    print("\nSaving model...")
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    print("\nEvaluating...")
    results, preds = evaluate(model, X_test, y_test, test)
    results["train_size"] = len(train)

    print(f"\n{'='*40}")
    print(f"RESULTS")
    print(f"{'='*40}")
    print(f"MAE:        {results['mae']}")
    print(f"RMSE:       {results['rmse']}")
    print(f"Within ±1:  {results['within_1']:.1%}")
    print(f"Within ±3:  {results['within_3']:.1%}")
    print(f"\nBy position:")
    for pos, pos_data in results["by_position"].items():
        print(f"  {pos:4s}: MAE {pos_data['mae']:.4f} ({pos_data['count']} samples)")

    # Save results
    results_path = os.path.join(EVAL_DIR, "xgboost_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Charts
    plot_feature_importance(model, os.path.join(EVAL_DIR, "xgboost_feature_importance.png"))
    plot_predictions_vs_actual(y_test, preds, os.path.join(EVAL_DIR, "xgboost_predictions.png"))


if __name__ == "__main__":
    main()
