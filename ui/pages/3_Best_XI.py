"""
Best Predicted XI — optimal 11 players shown on a pitch view.

Picks the highest-value starting XI within a 100m budget,
respecting position and team constraints (max 3 per team).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd

from ui.data_loader import get_enriched_predictions, load_fixture_lookahead
from ui.components import (
    metric_card_compact_html,
    pitch_player_html,
    pitch_html,
    fdr_legend_html,
)

# Valid formations (DEF, MID, FWD)
VALID_FORMATIONS = [
    (3, 5, 2), (3, 4, 3), (4, 4, 2), (4, 3, 3), (4, 5, 1),
    (5, 4, 1), (5, 3, 2), (5, 2, 3),
]

BUDGET = 100.0
MAX_PER_TEAM = 3


def pick_best_xi(df):
    """
    Pick the best predicted XI within budget and team constraints.

    Uses a greedy approach per formation: for each position, pick players
    sorted by combined_value_score, skipping any that would break the
    team limit or blow the budget. Returns the best formation found.
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
                "combined_value_score", ascending=False
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
        score = xi_df["combined_value_score"].sum()
        if score > best_score:
            best_score = score
            best_xi = xi_df

    return best_xi


def run():
    df = get_enriched_predictions()

    if df is None:
        st.error("No predictions found. Run `python models/predict_next_gw.py` first.")
        return

    gw = int(df["gameweek"].iloc[0])
    lookahead = load_fixture_lookahead(gw, num_gws=2)

    # ---- Header ----
    st.markdown(f"""
<div class="fpl-header">
<h1>Best Predicted XI</h1>
<p>Optimal Gameweek {gw} Starting XI &mdash; {BUDGET:.0f}m Budget</p>
</div>
""", unsafe_allow_html=True)

    # ---- Pick best XI ----
    xi = pick_best_xi(df)

    if xi is None or xi.empty:
        st.warning("Could not find a valid XI within budget constraints.")
        return

    # Captain = highest combined_value_score, Vice = second highest
    xi_sorted = xi.sort_values("combined_value_score", ascending=False)
    captain_id = xi_sorted.iloc[0]["player_id"]
    vice_captain_id = xi_sorted.iloc[1]["player_id"]

    captain_name = xi_sorted.iloc[0]["player_name"]
    vice_captain_name = xi_sorted.iloc[1]["player_name"]

    # Formation string
    pos_counts = xi["position"].value_counts()
    formation = f"{pos_counts.get('DEF', 0)}-{pos_counts.get('MID', 0)}-{pos_counts.get('FWD', 0)}"

    # Stats
    pts_cols = [c for c in ["xgb_predicted_pts", "llm_predicted_pts", "auto_xgb_predicted_pts", "claude_predicted_pts"] if c in xi.columns and xi[c].notna().any()]
    total_pts = xi[pts_cols].mean(axis=1).sum() if pts_cols else 0
    total_cost = xi["price"].sum()
    total_value = xi["combined_value_score"].sum()

    # ---- Info cards ----
    info_html = (
        f'<div class="mc-row">'
        f'{metric_card_compact_html("Formation", formation)}'
        f'{metric_card_compact_html("Total Pts", f"{total_pts:.1f}")}'
        f'{metric_card_compact_html("Total Value", f"{total_value:.1f}")}'
        f'{metric_card_compact_html("Cost", f"{total_cost:.1f}m")}'
        f'{metric_card_compact_html("Captain", captain_name)}'
        f'{metric_card_compact_html("Vice Captain", vice_captain_name)}'
        f'</div>'
    )
    st.markdown(info_html, unsafe_allow_html=True)

    # ---- Pitch view ----
    position_rows = []
    for pos in ["FWD", "MID", "DEF", "GK"]:
        pos_players = xi[xi["position"] == pos].sort_values(
            "combined_value_score", ascending=False
        )
        if len(pos_players) == 0:
            continue
        cards = []
        for _, player in pos_players.iterrows():
            team_id = int(player["team"])
            fixtures = lookahead.get(team_id, [])
            pid = player["player_id"]
            badge = "C" if pid == captain_id else ("V" if pid == vice_captain_id else "")
            cards.append(
                pitch_player_html(
                    player["player_name"],
                    player["team_name"],
                    player["price"],
                    player["combined_value_score"],
                    fixtures,
                    photo_url=player.get("photo_url"),
                    badge=badge,
                )
            )
        position_rows.append((pos, cards))

    st.markdown(pitch_html(position_rows), unsafe_allow_html=True)

    # ---- FDR Legend ----
    st.markdown(fdr_legend_html(), unsafe_allow_html=True)

    # ---- Squad table ----
    with st.expander("Full XI Details", expanded=False):
        table_cols = ["player_name", "position", "team_name", "price", "combined_value_score"]
        display_names = ["Player", "Pos", "Team", "Price", "Value"]

        for col_name, display in [
            ("xgb_predicted_pts", "XGB Pts"),
            ("auto_xgb_predicted_pts", "Auto Pts"),
            ("llm_predicted_pts", "LLM Pts"),
            ("claude_predicted_pts", "Claude Pts"),
        ]:
            if col_name in xi.columns and xi[col_name].notna().any():
                table_cols.append(col_name)
                display_names.append(display)

        details = xi[table_cols].copy()
        details.columns = display_names
        details["Role"] = details["Player"].apply(
            lambda n: "Captain" if n == captain_name else ("Vice" if n == vice_captain_name else "")
        )
        # Move Role to second column
        cols_order = ["Player", "Role"] + [c for c in details.columns if c not in ("Player", "Role")]
        details = details[cols_order].sort_values("Value", ascending=False)

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn("Price", format="£%.1f"),
                "Value": st.column_config.NumberColumn("Value", format="%.2f"),
            },
        )

    # ---- Footer ----
    st.markdown(
        '<div class="fpl-footer">'
        "Best XI is selected by highest Combined Value Score within a "
        f"{BUDGET:.0f}m budget, max {MAX_PER_TEAM} players per team"
        "</div>",
        unsafe_allow_html=True,
    )


run()
