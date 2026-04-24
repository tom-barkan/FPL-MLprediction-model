import sys, json, time, numpy as np, pandas as pd
from pathlib import Path
import importlib.util

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "processed" / "features.csv"
BASELINE_PATH = Path(__file__).parent / "baseline_mae.json"
TEMPORAL_SPLIT_GW = 34

def load_data():
    df = pd.read_csv(DATA_PATH)
    all_possible = [
        "form_3gw","form_5gw","avg_minutes_3gw","bonus_avg_3gw",
        "ict_index_3gw","goals_per_90_season","assists_per_90_season",
        "clean_sheets_pct_season","was_home","fixture_difficulty",
        "team_goals_scored_3gw","team_goals_conceded_3gw",
        "opponent_goals_conceded_season"
    ]
    feature_cols = [c for c in all_possible if c in df.columns]
    train = df[df["gameweek"] < TEMPORAL_SPLIT_GW]
    test  = df[df["gameweek"] >= TEMPORAL_SPLIT_GW]
    return (train[feature_cols].fillna(0), train["target_points"],
            test[feature_cols].fillna(0),  test["target_points"], feature_cols)

def evaluate(y_true, y_pred):
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    return {"mae": round(mae,4), "rmse": round(rmse,4),
            "within_1_pct": round(float(np.mean(np.abs(y_true-y_pred)<=1)*100),1),
            "within_3_pct": round(float(np.mean(np.abs(y_true-y_pred)<=3)*100),1)}

def load_baseline():
    return json.loads(BASELINE_PATH.read_text())["mae"] if BASELINE_PATH.exists() else 9999.0

def save_baseline(metrics):
    BASELINE_PATH.write_text(json.dumps(metrics, indent=2))
