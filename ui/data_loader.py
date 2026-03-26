"""
Centralized data loading with caching for the FPL dashboard.
"""

import json
import os

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_PATH = os.path.join(ROOT, "data", "processed", "next_gw_predictions.csv")
FIXTURES_PATH = os.path.join(ROOT, "data", "raw", "fixtures.json")
TEAMS_PATH = os.path.join(ROOT, "data", "raw", "teams.json")
PLAYERS_PATH = os.path.join(ROOT, "data", "raw", "players.json")


@st.cache_data
def load_predictions():
    """Load the next-GW predictions CSV."""
    if not os.path.exists(PREDICTIONS_PATH):
        return None
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data
def load_teams():
    """Load team ID -> info mapping."""
    with open(TEAMS_PATH) as f:
        teams = json.load(f)
    return {t["id"]: {"name": t["name"], "short_name": t["short_name"]} for t in teams}


@st.cache_data
def load_players_metadata():
    """Load player metadata from players.json (ownership, status, etc.)."""
    with open(PLAYERS_PATH) as f:
        players = json.load(f)
    return {
        p["id"]: {
            "web_name": p["web_name"],
            "selected_by_percent": float(p.get("selected_by_percent", 0)),
            "element_type": p["element_type"],
            "team": p["team"],
            "now_cost": p["now_cost"],
        }
        for p in players
    }


@st.cache_data
def load_fixture_lookahead(current_gw, num_gws=2):
    """
    Get upcoming fixtures for each team for the next `num_gws` gameweeks.

    Returns: {team_id: [{gw, opponent, home_away, fdr}, ...]}
    """
    with open(FIXTURES_PATH) as f:
        fixtures = json.load(f)

    teams = load_teams()
    target_gws = list(range(current_gw, current_gw + num_gws))
    lookahead = {}

    for fix in fixtures:
        event = fix.get("event")
        if event not in target_gws:
            continue

        team_h = fix["team_h"]
        team_a = fix["team_a"]

        # Home team entry
        lookahead.setdefault(team_h, []).append({
            "gw": event,
            "opponent": teams.get(team_a, {}).get("short_name", "???"),
            "home_away": "H",
            "fdr": fix.get("team_h_difficulty", 3),
        })

        # Away team entry
        lookahead.setdefault(team_a, []).append({
            "gw": event,
            "opponent": teams.get(team_h, {}).get("short_name", "???"),
            "home_away": "A",
            "fdr": fix.get("team_a_difficulty", 3),
        })

    # Sort each team's fixtures by GW
    for team_id in lookahead:
        lookahead[team_id].sort(key=lambda x: x["gw"])

    return lookahead


@st.cache_data
def get_enriched_predictions():
    """
    Load predictions and merge with fixture lookahead data.

    Adds columns for GW+1 fixture: gw2_opponent, gw2_home_away, gw2_fdr.
    Also adds ownership data from players.json.
    """
    df = load_predictions()
    if df is None:
        return None

    current_gw = int(df["gameweek"].iloc[0])
    lookahead = load_fixture_lookahead(current_gw, num_gws=2)
    players_meta = load_players_metadata()

    # Add GW+1 fixture columns
    gw2_opponents = []
    gw2_home_away = []
    gw2_fdr = []

    for _, row in df.iterrows():
        team_id = int(row["team"])
        team_fixtures = lookahead.get(team_id, [])

        # Find GW+1 fixture (second fixture in lookahead)
        gw2_fix = None
        for fix in team_fixtures:
            if fix["gw"] == current_gw + 1:
                gw2_fix = fix
                break

        if gw2_fix:
            gw2_opponents.append(gw2_fix["opponent"])
            gw2_home_away.append(gw2_fix["home_away"])
            gw2_fdr.append(gw2_fix["fdr"])
        else:
            gw2_opponents.append("-")
            gw2_home_away.append("-")
            gw2_fdr.append(0)

    df["gw2_opponent"] = gw2_opponents
    df["gw2_home_away"] = gw2_home_away
    df["gw2_fdr"] = gw2_fdr

    # Add ownership data
    df["ownership"] = df["player_id"].map(
        lambda pid: players_meta.get(pid, {}).get("selected_by_percent", 0)
    )

    return df
