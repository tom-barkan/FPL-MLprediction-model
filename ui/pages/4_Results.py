"""
Gameweek Results — compare model predictions against actual FPL points.

Shows per-model Best XI with actual points, captain/vice-captain scoring,
metrics comparison, and cumulative performance across gameweeks.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd

from ui.data_loader import (
    load_archived_predictions,
    get_available_result_gameweeks,
    load_gw_actual_points,
    load_players_metadata,
    PREDICTIONS_ARCHIVE_DIR,
    PHOTOS_DIR,
    PLAYER_PHOTO_URL_LOCAL,
)
from ui.results_logic import (
    evaluate_gameweek,
    compute_cumulative_results,
    MODEL_CONFIGS,
)
from ui.components import (
    metric_card_compact_html,
    pitch_player_results_html,
    pitch_html,
)


def _photo_url(player_id, players_meta):
    """Get local photo URL for a player."""
    code = players_meta.get(player_id, {}).get("code", 0)
    local_path = os.path.join(PHOTOS_DIR, f"p{code}.png")
    if os.path.exists(local_path):
        return PLAYER_PHOTO_URL_LOCAL.format(code=code)
    return None


def _render_scoreboard(gw_results):
    """Render the top-level scoreboard with total points per model."""
    if not gw_results:
        return

    # Find winner (highest actual points)
    best_model = max(gw_results, key=lambda m: gw_results[m]["scored"]["total_actual_pts"])

    cards_html = '<div class="mc-row">'
    for model_name, result in gw_results.items():
        pts = result["scored"]["total_actual_pts"]
        is_winner = model_name == best_model
        extra_class = ' class="results-winner"' if is_winner else ""
        trophy = " *" if is_winner else ""
        cards_html += (
            f'<div class="mc-item"{extra_class}>'
            f'<div class="mc-label">{model_name}</div>'
            f'<div class="mc-value">{pts} pts{trophy}</div>'
            f'</div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)


def _render_metrics_table(gw_results):
    """Render a comparison table of metrics across all models."""
    if not gw_results:
        return

    models = list(gw_results.keys())
    metrics = {m: gw_results[m]["metrics"] for m in models}

    # Find best values for highlighting
    best_actual = max(metrics[m]["total_actual"] for m in models)
    best_mae = min(metrics[m]["mae"] for m in models)
    best_delta = min(abs(metrics[m]["delta"]) for m in models)

    header = "<tr><th>Metric</th>" + "".join(f"<th>{m}</th>" for m in models) + "</tr>"

    def _cell(val, is_best, fmt=".1f"):
        cls = ' class="best-val"' if is_best else ""
        return f"<td{cls}>{val:{fmt}}</td>"

    row_actual = "<tr><td><b>Actual Points</b></td>"
    row_pred = "<tr><td>Predicted Points</td>"
    row_delta = "<tr><td>Delta (Pred - Actual)</td>"
    row_mae = "<tr><td>MAE (per player)</td>"

    for m in models:
        mx = metrics[m]
        row_actual += _cell(mx["total_actual"], mx["total_actual"] == best_actual, ".0f")
        row_pred += f"<td>{mx['total_predicted']:.1f}</td>"
        row_delta += _cell(abs(mx["delta"]), abs(mx["delta"]) == best_delta)
        row_mae += _cell(mx["mae"], mx["mae"] == best_mae, ".2f")

    row_actual += "</tr>"
    row_pred += "</tr>"
    row_delta += "</tr>"
    row_mae += "</tr>"

    table_html = (
        f'<table class="results-table">'
        f'{header}{row_actual}{row_pred}{row_delta}{row_mae}'
        f'</table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def _render_model_pitch(model_name, result, players_meta, is_winner=False):
    """Render a single model's Best XI as a pitch view with actual points."""
    xi = result["xi"]
    scored = result["scored"]
    captain_id = result["captain_id"]
    vice_id = result["vice_id"]

    # Build lookup for actual/predicted per player
    player_data = {p["player_id"]: p for p in scored["players"]}

    # Formation string
    pos_counts = xi["position"].value_counts()
    formation = f"{pos_counts.get('DEF', 0)}-{pos_counts.get('MID', 0)}-{pos_counts.get('FWD', 0)}"

    winner_class = " winner" if is_winner else ""
    trophy = " (Winner)" if is_winner else ""
    st.markdown(
        f'<div class="results-model-title{winner_class}">'
        f'{model_name}{trophy} &mdash; {formation} &mdash; '
        f'{scored["total_actual_pts"]} pts</div>',
        unsafe_allow_html=True,
    )

    # Build pitch rows
    position_rows = []
    for pos in ["FWD", "MID", "DEF", "GK"]:
        cfg = MODEL_CONFIGS.get(model_name, {})
        value_col = cfg.get("value_col", "combined_value_score")
        pos_players = xi[xi["position"] == pos]
        if value_col in pos_players.columns:
            pos_players = pos_players.sort_values(value_col, ascending=False)

        if len(pos_players) == 0:
            continue

        cards = []
        for _, player in pos_players.iterrows():
            pid = int(player["player_id"])
            pdata = player_data.get(pid, {})
            actual = pdata.get("effective_pts", pdata.get("actual_pts", 0))
            predicted = pdata.get("predicted_pts", 0)
            badge = "C" if pid == captain_id else ("V" if pid == vice_id else "")
            photo = _photo_url(pid, players_meta)

            cards.append(
                pitch_player_results_html(
                    player["player_name"],
                    player["team_name"],
                    player["price"],
                    predicted,
                    actual,
                    photo_url=photo,
                    badge=badge,
                )
            )
        position_rows.append((pos, cards))

    st.markdown(pitch_html(position_rows), unsafe_allow_html=True)


def _render_detail_table(gw_results):
    """Render detailed per-player comparison across all models."""
    all_players = {}
    models = list(gw_results.keys())

    for model_name, result in gw_results.items():
        for p in result["scored"]["players"]:
            pid = p["player_id"]
            if pid not in all_players:
                all_players[pid] = {
                    "Player": p["player_name"],
                    "Pos": p["position"],
                    "Team": p["team_name"],
                }
            role = ""
            if p["is_captain"]:
                role = "C"
            elif p["is_vice"]:
                role = "V"
            all_players[pid][f"{model_name} Pred"] = p["predicted_pts"]
            all_players[pid][f"{model_name} Actual"] = p["effective_pts"]
            all_players[pid][f"{model_name} Role"] = role

    if not all_players:
        return

    df = pd.DataFrame(all_players.values())
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_cumulative(available_gws):
    """Render cumulative performance across all available gameweeks."""
    if len(available_gws) < 1:
        return

    all_gw_results = {}
    for gw in sorted(available_gws):
        preds = load_archived_predictions(gw)
        actuals = load_gw_actual_points(gw)
        if preds is not None and actuals:
            all_gw_results[gw] = evaluate_gameweek(preds, actuals)

    if not all_gw_results:
        return

    cum_df = compute_cumulative_results(all_gw_results)
    if cum_df.empty:
        return

    st.markdown("---")
    st.markdown("""
<div class="fpl-header" style="padding:20px 24px;">
<h1 style="font-size:1.6em;">Cumulative Model Performance</h1>
<p>Running totals across all evaluated gameweeks</p>
</div>
""", unsafe_allow_html=True)

    # Summary cards — latest cumulative per model
    latest = cum_df.loc[cum_df.groupby("Model")["GW"].idxmax()]
    best_cum_model = latest.loc[latest["Cumulative Points"].idxmax(), "Model"]

    cards_html = '<div class="mc-row">'
    for _, row in latest.iterrows():
        is_best = row["Model"] == best_cum_model
        extra = ' class="results-winner"' if is_best else ""
        cards_html += (
            f'<div class="mc-item"{extra}>'
            f'<div class="mc-label">{row["Model"]}</div>'
            f'<div class="mc-value">{int(row["Cumulative Points"])} pts</div>'
            f'</div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # Line chart if multiple GWs
    if len(available_gws) > 1:
        chart_data = cum_df.pivot(index="GW", columns="Model", values="Cumulative Points")
        st.line_chart(chart_data)

    # Summary table
    summary_rows = []
    for _, row in latest.iterrows():
        summary_rows.append({
            "Model": row["Model"],
            "Total Points": int(row["Cumulative Points"]),
            "Avg MAE": row["Avg MAE"],
            "Gameweeks": int(row["GW"]) if len(available_gws) == 1 else len(available_gws),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("Total Points", ascending=False)
    summary_df["Rank"] = range(1, len(summary_df) + 1)
    summary_df = summary_df[["Rank", "Model", "Total Points", "Avg MAE", "Gameweeks"]]
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def run():
    # ---- Header ----
    st.markdown("""
<div class="fpl-header">
<h1>Gameweek Results</h1>
<p>How did each prediction model perform against reality?</p>
</div>
""", unsafe_allow_html=True)

    # ---- Check for archived predictions ----
    if not os.path.isdir(PREDICTIONS_ARCHIVE_DIR):
        st.info(
            "No archived predictions found. "
            "Run `python data/archive_predictions.py` to archive the current gameweek predictions."
        )
        return

    available_gws = get_available_result_gameweeks()

    if not available_gws:
        st.info(
            "No results available yet. Make sure you have:\n"
            "1. Archived predictions (`python data/archive_predictions.py`)\n"
            "2. Fetched actual results (`python data/fetch_fpl.py`)"
        )
        return

    # ---- GW Selector ----
    selected_gw = st.selectbox(
        "Select Gameweek",
        options=available_gws,
        format_func=lambda gw: f"Gameweek {gw}",
    )

    # ---- Load data ----
    predictions = load_archived_predictions(selected_gw)
    actuals = load_gw_actual_points(selected_gw)
    players_meta = load_players_metadata()

    if predictions is None:
        st.error(f"Could not load predictions for GW{selected_gw}")
        return

    if not actuals:
        st.warning(f"No actual results found for GW{selected_gw}. Run `python data/fetch_fpl.py` to update.")
        return

    # ---- Evaluate ----
    gw_results = evaluate_gameweek(predictions, actuals)

    if not gw_results:
        st.error("Could not evaluate any model for this gameweek.")
        return

    # ---- Scoreboard ----
    st.markdown('<div class="section-header">Model Scoreboard</div>', unsafe_allow_html=True)
    _render_scoreboard(gw_results)

    # ---- Metrics comparison ----
    st.markdown('<div class="section-header">Metrics Comparison</div>', unsafe_allow_html=True)
    _render_metrics_table(gw_results)

    # ---- Per-model pitch views ----
    st.markdown('<div class="section-header">Model Best XIs (Actual Points)</div>', unsafe_allow_html=True)

    best_model = max(gw_results, key=lambda m: gw_results[m]["scored"]["total_actual_pts"])
    models = list(gw_results.keys())

    # Display 2 per row
    for i in range(0, len(models), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(models):
                break
            model_name = models[idx]
            with col:
                _render_model_pitch(
                    model_name,
                    gw_results[model_name],
                    players_meta,
                    is_winner=(model_name == best_model),
                )

    # ---- Detail table ----
    with st.expander("Detailed Player Comparison", expanded=False):
        _render_detail_table(gw_results)

    # ---- Cumulative results ----
    _render_cumulative(available_gws)

    # ---- Footer ----
    st.markdown(
        '<div class="fpl-footer">'
        "Captain gets 2x points. Vice-captain gets 2x only if captain plays 0 minutes. "
        "Best XI selected per model using each model's own value scores."
        "</div>",
        unsafe_allow_html=True,
    )


run()
