"""
Computation logic for Gameweek Results analysis.

Pure functions — no Streamlit imports. Used by the Results page to
reconstruct each model's Best XI and score it against actual FPL points.
"""

import pandas as pd

MODEL_CONFIGS = {
    "XGBoost": {"value_col": "xgb_value_score", "pts_col": "xgb_predicted_pts"},
    "Auto-Research": {"value_col": "auto_xgb_value_score", "pts_col": "auto_xgb_predicted_pts"},
    "LLM": {"value_col": "llm_value_score", "pts_col": "llm_predicted_pts"},
    "Claude": {"value_col": "claude_value_score", "pts_col": "claude_predicted_pts"},
}

VALID_FORMATIONS = [
    (3, 5, 2), (3, 4, 3), (4, 4, 2), (4, 3, 3),
    (4, 5, 1), (5, 4, 1), (5, 3, 2), (5, 2, 3),
]

BUDGET = 100.0
MAX_PER_TEAM = 3


def get_available_models(df):
    """Return list of model names that have valid data in the DataFrame."""
    available = []
    for name, cfg in MODEL_CONFIGS.items():
        if cfg["pts_col"] in df.columns and df[cfg["pts_col"]].notna().any():
            available.append(name)
    return available


def pick_model_xi(df, value_col):
    """Pick the best XI using a specific model's value_score column.

    Same greedy algorithm as Best XI page: tries all valid formations,
    picks highest value_col players per position within budget/team constraints.
    """
    best_xi = None
    best_score = -1

    for n_def, n_mid, n_fwd in VALID_FORMATIONS:
        squad = []
        team_counts = {}
        budget_left = BUDGET

        position_needs = [
            ("GK", 1),
            ("DEF", n_def),
            ("MID", n_mid),
            ("FWD", n_fwd),
        ]

        failed = False
        for pos, count in position_needs:
            pos_players = df[df["position"] == pos].sort_values(
                value_col, ascending=False
            )
            picked = 0
            for _, player in pos_players.iterrows():
                if picked >= count:
                    break
                team = player["team_name"]
                price = player["price"]

                if team_counts.get(team, 0) >= MAX_PER_TEAM:
                    continue
                if price > budget_left:
                    continue

                squad.append(player)
                team_counts[team] = team_counts.get(team, 0) + 1
                budget_left -= price
                picked += 1

            if picked < count:
                failed = True
                break

        if failed or len(squad) != 11:
            continue

        xi_df = pd.DataFrame(squad)
        score = xi_df[value_col].sum()
        if score > best_score:
            best_score = score
            best_xi = xi_df

    return best_xi


def assign_captain_vice(xi, value_col):
    """Return (captain_player_id, vice_captain_player_id) based on model's value_col."""
    xi_sorted = xi.sort_values(value_col, ascending=False)
    captain_id = int(xi_sorted.iloc[0]["player_id"])
    vice_id = int(xi_sorted.iloc[1]["player_id"])
    return captain_id, vice_id


def score_model_xi(xi, captain_id, vice_id, actuals, pts_col):
    """Score a model's XI against actual FPL points.

    Full FPL captain rule: captain gets 2x points.
    Vice-captain gets 2x ONLY if captain played 0 minutes.

    Returns dict with total_actual_pts, total_predicted_pts, and per-player details.
    """
    captain_minutes = actuals.get(captain_id, {}).get("minutes", 0)
    vice_gets_double = captain_minutes == 0

    players = []
    total_actual = 0
    total_predicted = 0

    for _, row in xi.iterrows():
        pid = int(row["player_id"])
        actual_data = actuals.get(pid, {"total_points": 0, "minutes": 0})
        actual_pts = actual_data["total_points"]
        predicted_pts = row[pts_col] if pd.notna(row.get(pts_col)) else 0

        is_captain = pid == captain_id
        is_vice = pid == vice_id

        if is_captain:
            effective_pts = actual_pts * 2
        elif is_vice and vice_gets_double:
            effective_pts = actual_pts * 2
        else:
            effective_pts = actual_pts

        total_actual += effective_pts
        total_predicted += predicted_pts

        players.append({
            "player_id": pid,
            "player_name": row["player_name"],
            "position": row["position"],
            "team_name": row["team_name"],
            "price": row["price"],
            "predicted_pts": round(predicted_pts, 1),
            "actual_pts": actual_pts,
            "effective_pts": effective_pts,
            "is_captain": is_captain,
            "is_vice": is_vice,
            "minutes": actual_data["minutes"],
        })

    return {
        "total_actual_pts": total_actual,
        "total_predicted_pts": round(total_predicted, 1),
        "players": players,
    }


def compute_model_metrics(scored_result):
    """Compute accuracy metrics for a scored model XI.

    Returns: {"mae": float, "total_actual": int, "total_predicted": float, "delta": float}
    """
    players = scored_result["players"]
    if not players:
        return {"mae": 0, "total_actual": 0, "total_predicted": 0, "delta": 0}

    errors = [abs(p["predicted_pts"] - p["actual_pts"]) for p in players]
    mae = sum(errors) / len(errors)

    return {
        "mae": round(mae, 2),
        "total_actual": scored_result["total_actual_pts"],
        "total_predicted": scored_result["total_predicted_pts"],
        "delta": round(scored_result["total_predicted_pts"] - scored_result["total_actual_pts"], 1),
    }


def evaluate_gameweek(predictions_df, actuals):
    """Run full evaluation for all available models on a gameweek.

    Returns: {model_name: {"xi": DataFrame, "captain_id": int, "vice_id": int,
              "scored": dict, "metrics": dict}}
    """
    models = get_available_models(predictions_df)
    results = {}

    for model_name in models:
        cfg = MODEL_CONFIGS[model_name]
        xi = pick_model_xi(predictions_df, cfg["value_col"])
        if xi is None or xi.empty:
            continue

        captain_id, vice_id = assign_captain_vice(xi, cfg["value_col"])
        scored = score_model_xi(xi, captain_id, vice_id, actuals, cfg["pts_col"])
        metrics = compute_model_metrics(scored)

        results[model_name] = {
            "xi": xi,
            "captain_id": captain_id,
            "vice_id": vice_id,
            "scored": scored,
            "metrics": metrics,
        }

    return results


def compute_cumulative_results(all_gw_results):
    """Compute cumulative points across multiple gameweeks.

    Args:
        all_gw_results: {gw_num: {model_name: result_dict}}

    Returns: DataFrame with columns: Model, GW, GW Points, Cumulative Points, Avg MAE
    """
    rows = []
    model_totals = {}

    for gw in sorted(all_gw_results.keys()):
        gw_data = all_gw_results[gw]
        for model_name, result in gw_data.items():
            if model_name not in model_totals:
                model_totals[model_name] = {"cum_pts": 0, "mae_sum": 0, "gw_count": 0}

            gw_pts = result["scored"]["total_actual_pts"]
            mae = result["metrics"]["mae"]
            model_totals[model_name]["cum_pts"] += gw_pts
            model_totals[model_name]["mae_sum"] += mae
            model_totals[model_name]["gw_count"] += 1

            rows.append({
                "Model": model_name,
                "GW": gw,
                "GW Points": gw_pts,
                "Cumulative Points": model_totals[model_name]["cum_pts"],
                "Avg MAE": round(
                    model_totals[model_name]["mae_sum"] / model_totals[model_name]["gw_count"], 2
                ),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()
