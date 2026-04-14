"""
Centralized data loading with caching for the FPL dashboard.
"""

import json
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_PATH = os.path.join(ROOT, "data", "processed", "next_gw_predictions.csv")
FIXTURES_PATH = os.path.join(ROOT, "data", "raw", "fixtures.json")
TEAMS_PATH = os.path.join(ROOT, "data", "raw", "teams.json")
PLAYERS_PATH = os.path.join(ROOT, "data", "raw", "players.json")
HISTORIES_DIR = os.path.join(ROOT, "data", "raw", "player_histories")
PREDICTIONS_ARCHIVE_DIR = os.path.join(ROOT, "data", "predictions")
GAMEWEEKS_PATH = os.path.join(ROOT, "data", "raw", "gameweeks.json")


@st.cache_data(ttl=60)
def load_predictions():
    """Load the next-GW predictions CSV. TTL=60s so new columns appear on refresh."""
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
            "code": p.get("code", 0),
        }
        for p in players
    }


PLAYER_PHOTO_URL_CDN = "https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png"
PLAYER_PHOTO_URL_LOCAL = "app/static/photos/p{code}.png"
PHOTOS_DIR = os.path.join(ROOT, "ui", "static", "photos")


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
def load_last_n_gw_points(next_gw, n=3):
    """
    Load the last N completed gameweek points for each player.

    Returns: {player_id: "7, 2, 5"} (most recent last)
    """
    target_gws = list(range(next_gw - n, next_gw))  # e.g. [29, 30, 31] for next_gw=32
    result = {}

    if not os.path.isdir(HISTORIES_DIR):
        return result

    for filename in os.listdir(HISTORIES_DIR):
        if not filename.endswith(".json"):
            continue
        player_id = int(filename.replace(".json", ""))
        filepath = os.path.join(HISTORIES_DIR, filename)
        with open(filepath) as f:
            data = json.load(f)

        history = data.get("history", []) if isinstance(data, dict) else data
        gw_pts = {}
        for entry in history:
            if entry["round"] in target_gws:
                gw_pts[entry["round"]] = entry["total_points"]

        if gw_pts:
            pts_list = [str(gw_pts.get(gw, "-")) for gw in target_gws]
            result[player_id] = ", ".join(pts_list)

    return result


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

    # Add last 3 GW points
    last3_map = load_last_n_gw_points(current_gw, n=3)
    df["last_3_pts"] = df["player_id"].map(
        lambda pid: last3_map.get(pid, "-")
    )

    # Add player photo URLs (local static files only, None if missing)
    def _photo_url(pid):
        code = players_meta.get(pid, {}).get("code", 0)
        local_path = os.path.join(PHOTOS_DIR, f"p{code}.png")
        if os.path.exists(local_path):
            return PLAYER_PHOTO_URL_LOCAL.format(code=code)
        return None

    df["photo_url"] = df["player_id"].map(_photo_url)

    return df


@st.cache_data
def load_archived_predictions(gw):
    """Load archived predictions for a specific gameweek."""
    path = os.path.join(PREDICTIONS_ARCHIVE_DIR, f"gw{gw}_predictions.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=60)
def get_available_result_gameweeks():
    """Get list of gameweeks that have both archived predictions and actual results."""
    if not os.path.isdir(PREDICTIONS_ARCHIVE_DIR):
        return []

    import re
    gw_numbers = []
    for fname in os.listdir(PREDICTIONS_ARCHIVE_DIR):
        m = re.match(r"gw(\d+)_predictions\.csv", fname)
        if m:
            gw_numbers.append(int(m.group(1)))

    if not gw_numbers or not os.path.isdir(HISTORIES_DIR):
        return []

    # Check which GWs have actual results by sampling a few player history files
    available = []
    sample_files = [f for f in os.listdir(HISTORIES_DIR) if f.endswith(".json")][:5]
    for gw in gw_numbers:
        has_actuals = False
        for fname in sample_files:
            filepath = os.path.join(HISTORIES_DIR, fname)
            with open(filepath) as f:
                data = json.load(f)
            history = data.get("history", []) if isinstance(data, dict) else data
            for entry in history:
                if entry["round"] == gw:
                    has_actuals = True
                    break
            if has_actuals:
                break
        if has_actuals:
            available.append(gw)

    return sorted(available, reverse=True)


@st.cache_data
def load_gw_actual_points(gw):
    """Load actual FPL points for all players in a specific gameweek.

    Returns: {player_id: {"total_points": int, "minutes": int}}
    """
    result = {}
    if not os.path.isdir(HISTORIES_DIR):
        return result

    for filename in os.listdir(HISTORIES_DIR):
        if not filename.endswith(".json"):
            continue
        player_id = int(filename.replace(".json", ""))
        filepath = os.path.join(HISTORIES_DIR, filename)
        with open(filepath) as f:
            data = json.load(f)

        history = data.get("history", []) if isinstance(data, dict) else data
        for entry in history:
            if entry["round"] == gw:
                result[player_id] = {
                    "total_points": entry["total_points"],
                    "minutes": entry["minutes"],
                }
                break

    return result


def refresh_fpl_data():
    """Re-fetch all FPL data from the API. Deletes cached histories to force refresh."""
    import shutil
    import subprocess

    # Delete cached player histories so they get re-fetched
    if os.path.isdir(HISTORIES_DIR):
        shutil.rmtree(HISTORIES_DIR)
    os.makedirs(HISTORIES_DIR, exist_ok=True)

    fetch_script = os.path.join(ROOT, "data", "fetch_fpl.py")
    # Use the same Python interpreter that's running this process
    python_exe = sys.executable
    subprocess.run([python_exe, fetch_script], check=True)

    # Clear all cached data
    load_gw_actual_points.clear()
    get_available_result_gameweeks.clear()
    load_teams.clear()
    load_players_metadata.clear()
