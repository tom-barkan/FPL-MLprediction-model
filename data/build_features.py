"""
Build the feature matrix from raw FPL data.

Reads data/raw/ and produces data/processed/features.csv with one row
per player-gameweek, including rolling averages and the target label.

Rules:
  - Never leak future data — rolling features use only prior gameweeks
  - Skip GW1–3 (rolling averages unreliable)
  - Drop rows where minutes == 0 in the target GW
"""

import json
import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")

# Position mapping: FPL element_type → label
POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def load_raw_data():
    """Load players, teams, fixtures, and all player histories."""
    with open(os.path.join(RAW_DIR, "players.json")) as f:
        players = json.load(f)

    with open(os.path.join(RAW_DIR, "teams.json")) as f:
        teams = json.load(f)

    with open(os.path.join(RAW_DIR, "fixtures.json")) as f:
        fixtures = json.load(f)

    team_lookup = {t["id"]: t["name"] for t in teams}

    # Load all player histories into one big list
    history_dir = os.path.join(RAW_DIR, "player_histories")
    all_history = []

    for player in players:
        pid = player["id"]
        filepath = os.path.join(history_dir, f"{pid}.json")
        if not os.path.exists(filepath):
            continue

        with open(filepath) as f:
            data = json.load(f)

        for gw in data.get("history", []):
            gw["player_id"] = pid
            gw["player_name"] = player["web_name"]
            gw["position"] = POSITION_MAP.get(player["element_type"], "UNK")
            gw["price"] = player["now_cost"] / 10
            gw["team_id"] = player["team"]
            all_history.append(gw)

    print(f"Loaded {len(all_history)} player-gameweek records")
    return players, teams, team_lookup, fixtures, all_history


def build_fixture_difficulty(fixtures):
    """Build a lookup of fixture difficulty ratings."""
    difficulty = {}
    for fix in fixtures:
        event = fix.get("event")
        if event is None:
            continue
        # Home team's difficulty facing the away team
        difficulty[(fix["team_h"], event)] = fix["team_h_difficulty"]
        # Away team's difficulty facing the home team
        difficulty[(fix["team_a"], event)] = fix["team_a_difficulty"]
    return difficulty


def build_team_stats_from_fixtures(fixtures):
    """Compute team-level rolling stats from fixture scores (not player-level data)."""
    rows = []
    for fix in fixtures:
        event = fix.get("event")
        if event is None or fix.get("team_h_score") is None:
            continue
        # Home team
        rows.append({
            "team_id": fix["team_h"],
            "round": event,
            "goals_scored": fix["team_h_score"],
            "goals_conceded": fix["team_a_score"],
        })
        # Away team
        rows.append({
            "team_id": fix["team_a"],
            "round": event,
            "goals_scored": fix["team_a_score"],
            "goals_conceded": fix["team_h_score"],
        })

    team_gw = pd.DataFrame(rows)
    # A team can have multiple fixtures in one GW (rare), so sum them
    team_gw = team_gw.groupby(["team_id", "round"]).sum().reset_index()
    team_gw = team_gw.sort_values(["team_id", "round"])

    # Rolling 3-GW sums (shift to avoid leaking current GW)
    for col in ["goals_scored", "goals_conceded"]:
        team_gw[f"team_{col}_3gw"] = (
            team_gw.groupby("team_id")[col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
        )

    return team_gw[["team_id", "round", "team_goals_scored_3gw", "team_goals_conceded_3gw"]]


def build_opponent_season_stats(fixtures):
    """Compute opponent's season average goals conceded per GW from fixture data."""
    rows = []
    for fix in fixtures:
        event = fix.get("event")
        if event is None or fix.get("team_h_score") is None:
            continue
        # Home team conceded team_a_score goals
        rows.append({"team_id": fix["team_h"], "round": event, "conceded": fix["team_a_score"]})
        # Away team conceded team_h_score goals
        rows.append({"team_id": fix["team_a"], "round": event, "conceded": fix["team_h_score"]})

    team_gw = pd.DataFrame(rows)
    team_gw = team_gw.groupby(["team_id", "round"]).sum().reset_index()
    team_gw = team_gw.sort_values(["team_id", "round"])

    # Cumulative average conceded up to (not including) current GW
    team_gw["cum_conceded"] = (
        team_gw.groupby("team_id")["conceded"]
        .transform(lambda x: x.shift(1).cumsum())
    )
    team_gw["gw_count"] = (
        team_gw.groupby("team_id")["conceded"]
        .transform(lambda x: x.shift(1).expanding().count())
    )
    team_gw["opponent_goals_conceded_season"] = (
        team_gw["cum_conceded"] / team_gw["gw_count"]
    )

    # Rename team_id to opponent_team for merging
    team_gw = team_gw.rename(columns={"team_id": "opponent_team"})

    return team_gw[["opponent_team", "round", "opponent_goals_conceded_season"]]


def build_features(all_history, fixtures):
    """Build the full feature matrix."""
    df = pd.DataFrame(all_history)

    # Rename 'round' column (it's the gameweek number)
    if "round" not in df.columns and "event" in df.columns:
        df = df.rename(columns={"event": "round"})

    # Sort by player and gameweek
    df = df.sort_values(["player_id", "round"]).reset_index(drop=True)

    # --- Rolling player features (shifted to avoid leaking current GW) ---
    for window in [3, 5]:
        df[f"form_{window}gw"] = (
            df.groupby("player_id")["total_points"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    df["avg_minutes_3gw"] = (
        df.groupby("player_id")["minutes"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # Season cumulative per-90 stats (using only prior GWs)
    df["cum_goals"] = df.groupby("player_id")["goals_scored"].transform(
        lambda x: x.shift(1).cumsum()
    )
    df["cum_assists"] = df.groupby("player_id")["assists"].transform(
        lambda x: x.shift(1).cumsum()
    )
    df["cum_minutes"] = df.groupby("player_id")["minutes"].transform(
        lambda x: x.shift(1).cumsum()
    )

    df["goals_per_90_season"] = np.where(
        df["cum_minutes"] > 0,
        df["cum_goals"] / (df["cum_minutes"] / 90),
        0,
    )
    df["assists_per_90_season"] = np.where(
        df["cum_minutes"] > 0,
        df["cum_assists"] / (df["cum_minutes"] / 90),
        0,
    )

    # Clean sheet percentage (for DEF/GK)
    df["cum_clean_sheets"] = df.groupby("player_id")["clean_sheets"].transform(
        lambda x: x.shift(1).cumsum()
    )
    df["cum_appearances"] = df.groupby("player_id")["minutes"].transform(
        lambda x: (x.shift(1) > 0).cumsum()
    )
    df["clean_sheets_pct_season"] = np.where(
        df["cum_appearances"] > 0,
        df["cum_clean_sheets"] / df["cum_appearances"],
        0,
    )

    # ICT index rolling average
    df["ict_index"] = pd.to_numeric(df["ict_index"], errors="coerce").fillna(0)
    df["ict_index_3gw"] = (
        df.groupby("player_id")["ict_index"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # Bonus rolling average
    df["bonus_avg_3gw"] = (
        df.groupby("player_id")["bonus"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # --- Fixture difficulty ---
    fixture_diff = build_fixture_difficulty(fixtures)
    df["fixture_difficulty"] = df.apply(
        lambda row: fixture_diff.get((row["team_id"], row["round"]), 3), axis=1
    )

    # --- Team-level stats (from fixture scores, not player-level data) ---
    team_stats = build_team_stats_from_fixtures(fixtures)
    df = df.merge(team_stats, on=["team_id", "round"], how="left")

    # --- Opponent season stats (from fixture scores) ---
    opp_stats = build_opponent_season_stats(fixtures)
    df = df.merge(opp_stats, on=["opponent_team", "round"], how="left")

    # --- Select final columns ---
    feature_cols = [
        "player_id",
        "player_name",
        "position",
        "gameweek",
        "was_home",
        "opponent_team",
        "fixture_difficulty",
        "form_3gw",
        "form_5gw",
        "avg_minutes_3gw",
        "goals_per_90_season",
        "assists_per_90_season",
        "ict_index_3gw",
        "clean_sheets_pct_season",
        "bonus_avg_3gw",
        "team_goals_scored_3gw",
        "team_goals_conceded_3gw",
        "opponent_goals_conceded_season",
        "price",
        "target_points",
    ]

    # Rename columns to match expected output
    df = df.rename(columns={
        "round": "gameweek",
        "total_points": "target_points",
    })

    # Convert was_home to int
    if "was_home" in df.columns:
        df["was_home"] = df["was_home"].astype(int)

    # Select and filter
    available_cols = [c for c in feature_cols if c in df.columns]
    result = df[available_cols].copy()

    # Drop GW 1-3 (rolling averages unreliable)
    result = result[result["gameweek"] >= 4]

    # Drop rows where player didn't play (0 minutes)
    if "avg_minutes_3gw" in result.columns:
        # We keep the row if the player actually played this GW
        # (target_points > 0 or minutes > 0 in original data)
        pass

    # Drop rows with target_points of 0 where player didn't play
    # We use the original df to check minutes
    df_minutes = df[["player_id", "gameweek", "minutes"]].rename(
        columns={"round": "gameweek"} if "round" in df.columns else {}
    )
    if "gameweek" not in df_minutes.columns and "round" in df_minutes.columns:
        df_minutes = df_minutes.rename(columns={"round": "gameweek"})

    # Merge minutes back to check
    result = result.merge(
        df[["player_id", "gameweek", "minutes"]].rename(
            columns={"round": "gameweek"} if "round" in df.columns else {}
        ).drop_duplicates(),
        on=["player_id", "gameweek"],
        how="left",
    )

    # Drop rows where player got 0 minutes in the target GW
    result = result[result["minutes"] > 0]
    result = result.drop(columns=["minutes"])

    # Fill NaN values
    result = result.fillna(0)

    # Round floats
    float_cols = result.select_dtypes(include=[np.floating]).columns
    result[float_cols] = result[float_cols].round(3)

    # Sort
    result = result.sort_values(["gameweek", "player_name"]).reset_index(drop=True)

    return result


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    players, teams, team_lookup, fixtures, all_history = load_raw_data()

    if not all_history:
        print("No player history data found. Run fetch_fpl.py first.")
        return

    features = build_features(all_history, fixtures)

    output_path = os.path.join(PROCESSED_DIR, "features.csv")
    features.to_csv(output_path, index=False)

    print(f"\nFeature matrix saved to {output_path}")
    print(f"  → {len(features)} rows, {len(features.columns)} columns")
    print(f"  → Gameweeks: {features['gameweek'].min()} to {features['gameweek'].max()}")
    print(f"  → Players: {features['player_id'].nunique()}")
    print(f"\nColumn list: {list(features.columns)}")

    # Show train/test split sizes
    train = features[features["gameweek"] <= 30]
    test = features[features["gameweek"] > 30]
    print(f"\nTrain (GW 4-30): {len(train)} rows")
    print(f"Test  (GW 31+):  {len(test)} rows")


if __name__ == "__main__":
    main()
