"""
Generate predictions for the next gameweek for all active players
using XGBoost, fine-tuned LLM, and Auto-Research XGBoost.

Outputs: data/processed/next_gw_predictions.csv
"""

import json
import os
import re
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from mlx_lm import load, generate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(ROOT, "data", "processed", "features.csv")
MODEL_PATH = os.path.join(ROOT, "models", "xgboost_fpl.json")
LLM_DIR = os.path.join(ROOT, "models", "llama-3.2-3b")
ADAPTER_DIR = os.path.join(ROOT, "models", "fpl-lora-adapter-v3")
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUTPUT_PATH = os.path.join(ROOT, "data", "processed", "next_gw_predictions.csv")

# Auto-Research model uses features from autoresearch/prepare.py
AUTO_RESEARCH_FEATURES = [
    "form_3gw", "form_5gw", "avg_minutes_3gw", "bonus_avg_3gw",
    "ict_index_3gw", "goals_per_90_season", "assists_per_90_season",
    "clean_sheets_pct_season", "was_home", "fixture_difficulty",
    "team_goals_scored_3gw", "team_goals_conceded_3gw",
    "opponent_goals_conceded_season",
]

DROP_COLS = ["player_id", "player_name", "target_points", "gameweek"]

SYSTEM_PROMPT = (
    "You are an FPL points predictor. Given player stats and fixture info, "
    "predict the total FPL points for the gameweek. Respond with only a single integer."
)

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def get_next_gameweek():
    """Find the next unfinished gameweek."""
    with open(os.path.join(RAW_DIR, "gameweeks.json")) as f:
        gws = json.load(f)
    for gw in gws:
        if not gw["finished"]:
            return gw["id"]
    return gws[-1]["id"] + 1


def get_next_gw_fixtures(next_gw):
    """Get fixture info for the next gameweek."""
    with open(os.path.join(RAW_DIR, "fixtures.json")) as f:
        fixtures = json.load(f)

    gw_fixtures = {}
    for fix in fixtures:
        if fix.get("event") == next_gw:
            # Home team
            gw_fixtures[fix["team_h"]] = {
                "opponent_team": fix["team_a"],
                "was_home": 1,
                "fixture_difficulty": fix["team_h_difficulty"],
            }
            # Away team
            gw_fixtures[fix["team_a"]] = {
                "opponent_team": fix["team_h"],
                "was_home": 0,
                "fixture_difficulty": fix["team_a_difficulty"],
            }
    return gw_fixtures


def build_next_gw_features(next_gw):
    """
    Build feature rows for the next gameweek using the latest available data.
    Uses rolling stats from the most recent gameweeks.
    """
    df = pd.read_csv(FEATURES_PATH)
    fixtures = get_next_gw_fixtures(next_gw)

    if not fixtures:
        print(f"No fixtures found for GW {next_gw}")
        return pd.DataFrame()

    # Load player metadata
    with open(os.path.join(RAW_DIR, "players.json")) as f:
        players = json.load(f)

    # Get rolling stats for each player from recent data
    # Use the last row for each player (their most recent GW) as basis
    latest = df.sort_values("gameweek").groupby("player_id").last().reset_index()

    # Also load histories for minutes check
    history_dir = os.path.join(RAW_DIR, "player_histories")

    rows = []
    for player in players:
        pid = player["id"]
        team_id = player["team"]

        # Skip if team not playing this GW
        if team_id not in fixtures:
            continue

        # Skip if player has no recent data
        player_data = latest[latest["player_id"] == pid]
        if player_data.empty:
            continue

        p = player_data.iloc[0]

        # Skip if player hasn't been playing recently (avg < 20 mins)
        if p.get("avg_minutes_3gw", 0) < 20:
            continue

        fix = fixtures[team_id]

        row = {
            "player_id": pid,
            "player_name": player["web_name"],
            "position": POSITION_MAP.get(player["element_type"], "UNK"),
            "team": team_id,
            "gameweek": next_gw,
            "was_home": fix["was_home"],
            "opponent_team": fix["opponent_team"],
            "fixture_difficulty": fix["fixture_difficulty"],
            "form_3gw": p["form_3gw"],
            "form_5gw": p["form_5gw"],
            "avg_minutes_3gw": p["avg_minutes_3gw"],
            "goals_per_90_season": p["goals_per_90_season"],
            "assists_per_90_season": p["assists_per_90_season"],
            "ict_index_3gw": p["ict_index_3gw"],
            "clean_sheets_pct_season": p["clean_sheets_pct_season"],
            "bonus_avg_3gw": p["bonus_avg_3gw"],
            "team_goals_scored_3gw": p["team_goals_scored_3gw"],
            "team_goals_conceded_3gw": p["team_goals_conceded_3gw"],
            "opponent_goals_conceded_season": p["opponent_goals_conceded_season"],
            "price": player["now_cost"] / 10,
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    print(f"Built {len(result)} player rows for GW {next_gw}")
    return result


def predict_xgboost(next_gw_df):
    """Run XGBoost predictions with confidence estimation."""
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    feature_cols = [c for c in next_gw_df.columns
                    if c not in ["player_id", "player_name", "gameweek", "team"]]
    X = next_gw_df[feature_cols].copy()
    X = pd.get_dummies(X, columns=["position"])

    # Align columns with training
    train_df = pd.read_csv(FEATURES_PATH)
    train = train_df[train_df["gameweek"] <= 30]
    X_train = train.drop(columns=DROP_COLS)
    X_train = pd.get_dummies(X_train, columns=["position"])
    X = X.reindex(columns=X_train.columns, fill_value=0)

    preds = model.predict(X)

    # Confidence: use prediction margin from ensemble (leaf predictions std)
    # Lower std across trees = higher confidence
    leaf_preds = model.get_booster().predict(
        xgb.DMatrix(X, feature_names=list(X.columns)),
        pred_leaf=True,
    )
    # Estimate confidence from prediction variance across tree subsets
    n_trees = leaf_preds.shape[1]
    chunk_size = max(1, n_trees // 5)
    chunk_preds = []
    for i in range(0, n_trees, chunk_size):
        chunk_end = min(i + chunk_size, n_trees)
        iteration_range = (0, chunk_end)
        chunk_pred = model.get_booster().predict(
            xgb.DMatrix(X, feature_names=list(X.columns)),
            iteration_range=iteration_range,
        )
        chunk_preds.append(chunk_pred)

    chunk_preds = np.array(chunk_preds)
    pred_std = np.std(chunk_preds, axis=0)

    # Normalize confidence: lower std = higher confidence (0-100 scale)
    max_std = np.percentile(pred_std, 95) if len(pred_std) > 0 else 1
    confidence = np.clip(100 * (1 - pred_std / max(max_std, 0.01)), 10, 99).astype(int)

    return preds, confidence


def predict_auto_research(next_gw_df):
    """
    Run Auto-Research XGBoost predictions.

    Trains the two-stage model (classifier + regressor) on all historical data
    using the optimised hyperparameters found by the autoresearch loop, then
    predicts for the next gameweek.
    """
    # Import autoresearch train module
    sys.path.insert(0, os.path.join(ROOT, "autoresearch"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "autoresearch_train", os.path.join(ROOT, "autoresearch", "train.py")
    )
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    # Load historical data for training
    hist = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in AUTO_RESEARCH_FEATURES if c in hist.columns]
    X_train = hist[feature_cols].fillna(0)
    y_train = hist["target_points"]

    # Train model on all historical data
    model = train_mod.build_and_train(X_train, y_train, feature_cols)

    # Prepare next-GW features (same columns)
    X_next = next_gw_df[feature_cols].fillna(0).copy()
    preds = train_mod.predict(model, X_next, feature_cols)

    # Confidence: use the classifier's play probability as a basis,
    # combined with form consistency for a richer signal.
    play_prob = model["clf"].predict_proba(X_next[model["_used_cols"]])[:, 1]

    conf = np.full(len(next_gw_df), 50.0)

    # Play probability signal: high prob = more confident
    conf += np.clip(play_prob * 30, 0, 30)

    # Form consistency
    if "form_3gw" in next_gw_df.columns and "form_5gw" in next_gw_df.columns:
        form_consistency = np.clip(
            15 - np.abs(next_gw_df["form_3gw"].values - next_gw_df["form_5gw"].values) * 5, 0, 15
        )
        conf += form_consistency

    # Minutes stability
    if "avg_minutes_3gw" in next_gw_df.columns:
        mins_factor = np.clip((next_gw_df["avg_minutes_3gw"].values - 45) / 45 * 10, 0, 10)
        conf += mins_factor

    confidence = np.clip(conf, 15, 95).astype(int)
    return preds, confidence


def predict_llm(next_gw_df):
    """Run fine-tuned LLM predictions with confidence estimation."""
    print("Loading fine-tuned LLM...")
    model, tokenizer = load(LLM_DIR, adapter_path=ADAPTER_DIR)

    preds = []

    for i, (_, row) in enumerate(next_gw_df.iterrows()):
        home_away = "Home" if row["was_home"] == 1 else "Away"
        user_content = (
            f"Player: {row['player_name']} ({row['position']})\n"
            f"Gameweek: {int(row['gameweek'])}\n"
            f"Fixture: {home_away} (FDR: {int(row['fixture_difficulty'])})\n"
            f"Form (last 3 GW avg): {row['form_3gw']:.1f} pts\n"
            f"Form (last 5 GW avg): {row['form_5gw']:.1f} pts\n"
            f"Minutes (last 3 GW avg): {row['avg_minutes_3gw']:.1f}\n"
            f"Goals per 90 (season): {row['goals_per_90_season']:.2f}\n"
            f"Assists per 90 (season): {row['assists_per_90_season']:.2f}\n"
            f"ICT index (last 3 GW avg): {row['ict_index_3gw']:.1f}\n"
            f"Clean sheet % (season): {row['clean_sheets_pct_season']:.1%}\n"
            f"Bonus avg (last 3 GW): {row['bonus_avg_3gw']:.1f}\n"
            f"Team goals scored (last 3 GW): {row['team_goals_scored_3gw']:.0f}\n"
            f"Team goals conceded (last 3 GW): {row['team_goals_conceded_3gw']:.0f}\n"
            f"Opponent goals conceded (season avg per GW): {row['opponent_goals_conceded_season']:.1f}\n"
            f"Price: {row['price']:.1f}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        resp = generate(model, tokenizer, prompt=prompt, max_tokens=10)
        parsed = _parse_int(resp)
        preds.append(parsed if parsed is not None else 2)

        if (i + 1) % 50 == 0:
            print(f"  LLM: {i+1}/{len(next_gw_df)} predictions done")

    del model, tokenizer
    raw_preds = np.array(preds, dtype=float)

    # Blend raw LLM prediction with player features for smoother output.
    # The LLM gives a directional signal (0/1/2/4/9/10), but we refine it
    # using form, ICT, and fixture data to produce realistic decimal values.
    preds = _smooth_predictions(raw_preds, next_gw_df)

    # Confidence heuristic based on feature clarity:
    # Higher confidence when: good form consistency (3gw ~= 5gw), high minutes,
    # clearer fixture (FDR 1-2 or 5), higher ICT signal
    confidences = _compute_feature_confidence(next_gw_df, preds)

    return preds, confidences


def _smooth_predictions(raw_preds, df):
    """
    Blend the LLM's integer prediction with player features to produce
    smoother, more realistic decimal predictions.

    The LLM gives a directional signal (low/mid/high), and we refine it:
    - 40% LLM signal (dampened toward realistic range)
    - 30% recent form (the most predictive feature)
    - 30% feature-based estimate (ICT, fixture, minutes)
    """
    form = df["form_3gw"].values
    form5 = df["form_5gw"].values
    ict = df["ict_index_3gw"].values
    fdr = df["fixture_difficulty"].values
    minutes = df["avg_minutes_3gw"].values
    goals_p90 = df["goals_per_90_season"].values
    assists_p90 = df["assists_per_90_season"].values
    cs_pct = df["clean_sheets_pct_season"].values
    positions = df["position"].values

    # 1. Dampen the LLM prediction toward the mean (reduce 9/10 overshoot)
    # Map raw LLM: compress toward 1-7 range
    llm_dampened = np.clip(raw_preds * 0.6 + 1.0, 0.5, 7.5)

    # 2. Feature-based estimate
    # Start from form, adjust for fixture and attacking threat
    fixture_adj = np.where(fdr <= 2, 0.5, np.where(fdr >= 4, -0.5, 0.0))
    attack_bonus = (goals_p90 + assists_p90) * 2.0
    cs_bonus = np.where(np.isin(positions, ["GK", "DEF"]), cs_pct * 3.0, 0.0)
    minutes_factor = np.clip(minutes / 90.0, 0.0, 1.0)

    feature_est = (form * 0.5 + form5 * 0.3 + ict * 0.15 +
                   fixture_adj + attack_bonus + cs_bonus) * minutes_factor

    # 3. Blend: 40% LLM, 30% form, 30% feature estimate
    blended = 0.40 * llm_dampened + 0.30 * form + 0.30 * feature_est

    # Clip to realistic FPL range and round to 1 decimal
    blended = np.clip(blended, 0.5, 10.0)
    blended = np.round(blended, 1)

    return blended


def _compute_feature_confidence(df, preds):
    """
    Estimate prediction confidence from input features.
    Higher when: consistent form, high minutes (nailed), clear fixture,
    prediction aligns with form.
    """
    conf = np.full(len(df), 50.0)

    form_3 = df["form_3gw"].values
    form_5 = df["form_5gw"].values
    minutes = df["avg_minutes_3gw"].values
    fdr = df["fixture_difficulty"].values
    ict = df["ict_index_3gw"].values

    # Form consistency: 3gw and 5gw averages are close
    form_consistency = np.clip(20 - np.abs(form_3 - form_5) * 8, 0, 20)
    conf += form_consistency

    # Minutes: nailed starters are more predictable
    minutes_factor = np.clip((minutes - 45) / 45 * 15, 0, 15)
    conf += minutes_factor

    # Fixture clarity: extreme FDR (1-2 easy, 5 hard) = more predictable
    fixture_clarity = np.where(
        (fdr <= 2) | (fdr >= 5), 10,
        np.where(fdr == 3, 0, 5)
    )
    conf += fixture_clarity

    # Prediction aligns with form: prediction close to form = more confident
    form_alignment = np.clip(15 - np.abs(preds - form_3) * 5, 0, 15)
    conf += form_alignment

    return np.clip(conf, 15, 95).astype(int)


def _parse_int(response):
    cleaned = response.strip()
    try:
        return int(cleaned)
    except ValueError:
        match = re.search(r'\b(\d+)\b', cleaned)
        if match:
            return int(match.group(1))
    return None


def compute_value_score(pred_points, confidence):
    """
    Combine predicted points and confidence into a single value score.
    Value = predicted_points * (confidence / 100)
    This rewards high-point predictions with high confidence.
    """
    return np.round(pred_points * (confidence / 100), 2)


def main():
    next_gw = get_next_gameweek()
    print(f"Predicting for Gameweek {next_gw}\n")

    # Build features for next GW
    next_gw_df = build_next_gw_features(next_gw)
    if next_gw_df.empty:
        print("No data to predict. Exiting.")
        return

    # Load team names
    with open(os.path.join(RAW_DIR, "teams.json")) as f:
        teams = json.load(f)
    team_names = {t["id"]: t["short_name"] for t in teams}

    # XGBoost predictions
    print("\nRunning XGBoost predictions...")
    xgb_preds, xgb_conf = predict_xgboost(next_gw_df)

    # Auto-Research XGBoost predictions
    print("\nRunning Auto-Research XGBoost predictions...")
    auto_preds, auto_conf = predict_auto_research(next_gw_df)

    # LLM predictions
    print("\nRunning LLM predictions...")
    llm_preds, llm_conf = predict_llm(next_gw_df)

    # Build output
    output = next_gw_df[["player_id", "player_name", "position", "team", "price",
                          "gameweek", "was_home", "fixture_difficulty",
                          "form_3gw", "ict_index_3gw", "avg_minutes_3gw"]].copy()

    output["team_name"] = output["team"].map(team_names)
    output["opponent"] = next_gw_df["opponent_team"].map(team_names)
    output["home_away"] = output["was_home"].map({1: "H", 0: "A"})

    # XGBoost
    output["xgb_predicted_pts"] = np.round(xgb_preds, 1)
    output["xgb_confidence"] = xgb_conf
    output["xgb_value_score"] = compute_value_score(xgb_preds, xgb_conf)

    # Auto-Research XGBoost
    output["auto_xgb_predicted_pts"] = np.round(auto_preds, 1)
    output["auto_xgb_confidence"] = auto_conf
    output["auto_xgb_value_score"] = compute_value_score(auto_preds, auto_conf)

    # LLM
    output["llm_predicted_pts"] = np.round(llm_preds, 1)
    output["llm_confidence"] = llm_conf
    output["llm_value_score"] = compute_value_score(llm_preds, llm_conf)

    # Combined value score (average of all three)
    output["combined_value_score"] = np.round(
        (output["xgb_value_score"] + output["auto_xgb_value_score"] + output["llm_value_score"]) / 3, 2
    )

    # Sort by combined value score
    output = output.sort_values("combined_value_score", ascending=False).reset_index(drop=True)

    # Save
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPredictions saved to {OUTPUT_PATH}")
    print(f"Total players predicted: {len(output)}")

    # Show top 15
    print(f"\n{'='*80}")
    print(f"TOP 15 PLAYERS — GW {next_gw}")
    print(f"{'='*80}")
    print(f"{'Player':<16} {'Pos':>4} {'Team':>5} {'Fix':>6} {'XGB':>5} {'Auto':>5} {'LLM':>5} {'Value':>6}")
    print(f"{'-'*80}")
    for _, r in output.head(15).iterrows():
        fix_str = f"{r['home_away']} {r['opponent']}"
        print(
            f"{r['player_name']:<16} {r['position']:>4} {r['team_name']:>5} "
            f"{fix_str:>6} {r['xgb_predicted_pts']:>5.1f} "
            f"{r['auto_xgb_predicted_pts']:>5.1f} "
            f"{r['llm_predicted_pts']:>5.1f} "
            f"{r['combined_value_score']:>6.2f}"
        )


if __name__ == "__main__":
    main()
