"""
Fetch all required data from the FPL API and save to data/raw/.

Endpoints used:
  - /api/bootstrap-static/  → players, teams, gameweeks
  - /api/element-summary/{id}/ → per-player gameweek history
  - /api/fixtures/ → all fixtures with scores and difficulty
"""

import json
import os
import time
import requests

BASE_URL = "https://fantasy.premierleague.com"
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def fetch_bootstrap():
    """Fetch the master dataset: all players, teams, gameweeks."""
    print("Fetching bootstrap-static...")
    resp = requests.get(f"{BASE_URL}/api/bootstrap-static/", timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Save full response
    with open(os.path.join(RAW_DIR, "bootstrap.json"), "w") as f:
        json.dump(data, f, indent=2)

    # Extract and save players
    players = data["elements"]
    with open(os.path.join(RAW_DIR, "players.json"), "w") as f:
        json.dump(players, f, indent=2)

    # Extract and save teams
    teams = data["teams"]
    with open(os.path.join(RAW_DIR, "teams.json"), "w") as f:
        json.dump(teams, f, indent=2)

    # Extract and save gameweeks (events)
    gameweeks = data["events"]
    with open(os.path.join(RAW_DIR, "gameweeks.json"), "w") as f:
        json.dump(gameweeks, f, indent=2)

    # Extract and save element types (positions)
    element_types = data["element_types"]
    with open(os.path.join(RAW_DIR, "element_types.json"), "w") as f:
        json.dump(element_types, f, indent=2)

    print(f"  → {len(players)} players, {len(teams)} teams, {len(gameweeks)} gameweeks")
    return players


def fetch_player_histories(players):
    """Fetch gameweek-by-gameweek history for every player."""
    history_dir = os.path.join(RAW_DIR, "player_histories")
    os.makedirs(history_dir, exist_ok=True)

    total = len(players)
    print(f"Fetching histories for {total} players...")

    for i, player in enumerate(players):
        player_id = player["id"]
        filepath = os.path.join(history_dir, f"{player_id}.json")

        # Skip if already fetched (allows resuming)
        if os.path.exists(filepath):
            if (i + 1) % 100 == 0:
                print(f"  → {i + 1}/{total} (cached)")
            continue

        try:
            resp = requests.get(
                f"{BASE_URL}/api/element-summary/{player_id}/",
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        except requests.RequestException as e:
            print(f"  ! Failed for player {player_id}: {e}")

        if (i + 1) % 50 == 0:
            print(f"  → {i + 1}/{total}")

        # Be polite — 1 second delay between requests
        time.sleep(1)

    print(f"  → Done. {total} player histories saved.")


def fetch_fixtures():
    """Fetch all fixtures with scores and difficulty ratings."""
    print("Fetching fixtures...")
    resp = requests.get(f"{BASE_URL}/api/fixtures/", timeout=30)
    resp.raise_for_status()
    fixtures = resp.json()

    with open(os.path.join(RAW_DIR, "fixtures.json"), "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"  → {len(fixtures)} fixtures")
    return fixtures


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(os.path.join(RAW_DIR, "player_histories"), exist_ok=True)

    players = fetch_bootstrap()
    fetch_fixtures()
    fetch_player_histories(players)

    print("\nAll data fetched and saved to data/raw/")


if __name__ == "__main__":
    main()
