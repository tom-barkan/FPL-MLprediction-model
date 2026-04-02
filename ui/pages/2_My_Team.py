"""
My Team Page — Squad builder + multi-transfer planner.

Two-column layout: pitch on left, transfer planner on right.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd

from ui.data_loader import get_enriched_predictions, load_fixture_lookahead

# Ensure session state is initialized (needed when navigating directly to this page)
if "squad" not in st.session_state:
    st.session_state.squad = []
if "budget" not in st.session_state:
    st.session_state.budget = 100.0
from ui.components import (
    fixture_badge_html,
    fixture_run_html,
    player_card_html,
    metric_card_html,
    metric_card_compact_html,
    delta_html,
    fdr_legend_html,
    pitch_player_html,
    pitch_html,
    suggestion_card_html,
)


# ---- Constants ----
POSITION_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3
TOTAL_BUDGET = 100.0

# Valid formations for starting XI (DEF, MID, FWD)
VALID_FORMATIONS = [
    (3, 5, 2), (3, 4, 3), (4, 4, 2), (4, 3, 3), (4, 5, 1),
    (5, 4, 1), (5, 3, 2), (5, 2, 3),
]


def get_squad_df(df):
    """Get dataframe filtered to current squad players."""
    squad_ids = st.session_state.squad
    if not squad_ids:
        return pd.DataFrame()
    return df[df["player_id"].isin(squad_ids)].copy()


def validate_squad(df, squad_ids):
    """Return list of validation warnings for the squad."""
    warnings = []
    squad = df[df["player_id"].isin(squad_ids)]

    # Position counts
    pos_counts = squad["position"].value_counts().to_dict()
    for pos, limit in POSITION_LIMITS.items():
        count = pos_counts.get(pos, 0)
        if count > limit:
            warnings.append(f"Too many {pos}: {count}/{limit}")

    # Team limit
    team_counts = squad["team_name"].value_counts()
    for team, count in team_counts.items():
        if count > MAX_PER_TEAM:
            warnings.append(f"Too many from {team}: {count}/{MAX_PER_TEAM}")

    # Budget
    total_cost = squad["price"].sum()
    if total_cost > TOTAL_BUDGET:
        warnings.append(f"Over budget: {total_cost:.1f}m / {TOTAL_BUDGET:.1f}m")

    return warnings


def auto_pick_xi(squad_df):
    """
    Auto-pick the best starting XI from 15 players.
    Returns (xi_df, captain_id).
    """
    if len(squad_df) < 11:
        return squad_df, squad_df["player_id"].iloc[0] if len(squad_df) > 0 else None

    squad_sorted = squad_df.sort_values("combined_value_score", ascending=False)
    gks = squad_sorted[squad_sorted["position"] == "GK"]
    defs = squad_sorted[squad_sorted["position"] == "DEF"]
    mids = squad_sorted[squad_sorted["position"] == "MID"]
    fwds = squad_sorted[squad_sorted["position"] == "FWD"]

    best_xi = None
    best_score = -1

    for n_def, n_mid, n_fwd in VALID_FORMATIONS:
        if len(defs) < n_def or len(mids) < n_mid or len(fwds) < n_fwd or len(gks) < 1:
            continue

        xi = pd.concat([
            gks.head(1),
            defs.head(n_def),
            mids.head(n_mid),
            fwds.head(n_fwd),
        ])

        score = xi["combined_value_score"].sum()
        if score > best_score:
            best_score = score
            best_xi = xi

    if best_xi is None:
        best_xi = squad_sorted.head(11)

    captain_id = best_xi.loc[best_xi["combined_value_score"].idxmax(), "player_id"]
    return best_xi, captain_id


def build_pitch_html(squad_df, lookahead):
    """Build the full pitch HTML with players arranged in formation rows."""
    # Order: FWD (top), MID, DEF, GK (bottom)
    position_rows = []
    for pos in ["FWD", "MID", "DEF", "GK"]:
        pos_players = squad_df[squad_df["position"] == pos].sort_values(
            "combined_value_score", ascending=False
        )
        if len(pos_players) == 0:
            continue
        cards = []
        for _, player in pos_players.iterrows():
            team_id = int(player["team"])
            fixtures = lookahead.get(team_id, [])
            cards.append(
                pitch_player_html(
                    player["player_name"],
                    player["team_name"],
                    player["price"],
                    player["combined_value_score"],
                    fixtures,
                    photo_url=player.get("photo_url"),
                )
            )
        position_rows.append((pos, cards))
    return pitch_html(position_rows)


def get_transfer_suggestions(df, squad_df, player_out, strategy, gw):
    """Get transfer-in suggestions based on strategy."""
    pos = player_out["position"]
    squad_ids = set(squad_df["player_id"].tolist())
    squad_teams = squad_df["team_name"].value_counts().to_dict()

    # Base filter: same position, not in squad
    candidates = df[
        (df["position"] == pos) & (~df["player_id"].isin(squad_ids))
    ].copy()

    # Budget filter: player_in price <= player_out price + remaining budget
    remaining_budget = TOTAL_BUDGET - squad_df["price"].sum()
    max_price = player_out["price"] + remaining_budget
    candidates = candidates[candidates["price"] <= max_price]

    # Team limit filter
    def team_ok(team_name):
        current = squad_teams.get(team_name, 0)
        # Remove outgoing player's team count
        if team_name == player_out["team_name"]:
            current -= 1
        return current < MAX_PER_TEAM

    candidates = candidates[candidates["team_name"].apply(team_ok)]

    if candidates.empty:
        return candidates

    # Apply strategy
    if strategy == "Safe":
        avg_conf = (candidates["xgb_confidence"] + candidates["llm_confidence"]) / 2
        candidates = candidates[avg_conf >= 65]
        candidates = candidates.sort_values("combined_value_score", ascending=False)

    elif strategy == "Differential":
        avg_pts = (candidates["xgb_predicted_pts"] + candidates["llm_predicted_pts"]) / 2
        avg_conf = (candidates["xgb_confidence"] + candidates["llm_confidence"]) / 2
        candidates = candidates[(avg_pts >= 3.5) & (avg_conf < 55)]
        candidates = candidates.sort_values("combined_value_score", ascending=False)

    elif strategy == "Form":
        candidates = candidates.sort_values(
            ["form_3gw", "avg_minutes_3gw"], ascending=[False, False]
        )

    elif strategy == "Fixture":
        # Average FDR across next 2 GWs (lower = easier)
        candidates["avg_fdr"] = (
            candidates["fixture_difficulty"].astype(float) + candidates["gw2_fdr"].astype(float)
        ) / 2
        candidates = candidates.sort_values("avg_fdr", ascending=True)

    return candidates.head(10)


def render_suggestion_card(player_in, player_out, lookahead, idx, transfer_idx):
    """Render a suggestion card with an Apply button below it."""
    team_id_in = int(player_in["team"])
    fixtures_in = lookahead.get(team_id_in, [])

    # Render card HTML
    st.markdown(
        suggestion_card_html(player_in, player_out, fixtures_in),
        unsafe_allow_html=True,
    )

    # Apply button (Streamlit widget -- must be outside HTML)
    if st.button(
        f"Apply: {player_in['player_name']}",
        key=f"apply_{transfer_idx}_{idx}",
        type="primary",
        use_container_width=True,
    ):
        squad = st.session_state.squad.copy()
        squad.remove(int(player_out["player_id"]))
        squad.append(int(player_in["player_id"]))
        st.session_state.squad = squad
        st.rerun()


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
        <h1>FPL Squad & Transfer Planner</h1>
        <p>Squad Builder & Transfer Advisor &mdash; Gameweek {gw}</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Squad Builder (player selector) ----
    player_options = {}
    for _, row in df.iterrows():
        label = f"{row['player_name']} - {row['position']} - {row['team_name']} - {row['price']:.1f}m"
        player_options[label] = int(row["player_id"])

    id_to_label = {v: k for k, v in player_options.items()}
    current_labels = [id_to_label[pid] for pid in st.session_state.squad if pid in id_to_label]

    # Sync session state → widget key before rendering
    st.session_state.squad_selector = current_labels

    def _on_squad_change():
        st.session_state.squad = [
            player_options[label] for label in st.session_state.squad_selector
        ]

    st.multiselect(
        "Select your 15 players",
        options=sorted(player_options.keys()),
        key="squad_selector",
        max_selections=15,
        help="Search by player name. Select up to 15 players (2 GK, 5 DEF, 5 MID, 3 FWD).",
        on_change=_on_squad_change,
    )

    squad_df = get_squad_df(df)

    if squad_df.empty:
        st.info("Select players above to build your squad.")
        return

    # ---- Validation ----
    warnings = validate_squad(df, st.session_state.squad)
    for w in warnings:
        st.warning(w)

    # ---- Header Metric Cards Row ----
    total_cost = squad_df["price"].sum()
    remaining = TOTAL_BUDGET - total_cost
    pos_counts = squad_df["position"].value_counts().to_dict()

    metrics_items = [
        ("Budget Left", f"{remaining:.1f}m"),
        ("Squad Composition GK", f"{pos_counts.get('GK', 0)}/{POSITION_LIMITS['GK']}"),
        ("DEF", f"{pos_counts.get('DEF', 0)}/{POSITION_LIMITS['DEF']}"),
        ("MID", f"{pos_counts.get('MID', 0)}/{POSITION_LIMITS['MID']}"),
        ("FWD", f"{pos_counts.get('FWD', 0)}/{POSITION_LIMITS['FWD']}"),
    ]
    mc_html = '<div class="mc-row">'
    for lbl, val in metrics_items:
        mc_html += metric_card_compact_html(lbl, val)
    mc_html += '</div>'
    st.markdown(mc_html, unsafe_allow_html=True)

    # ---- FDR Legend ----
    st.markdown(fdr_legend_html(), unsafe_allow_html=True)

    # ---- Two-Column Layout: Pitch + Transfer Planner ----
    left_col, right_col = st.columns([5, 6], gap="large")

    # ======== LEFT COLUMN: Pitch ========
    with left_col:
        st.markdown(
            '<span class="section-header">Your Squad</span>',
            unsafe_allow_html=True,
        )
        st.markdown(build_pitch_html(squad_df, lookahead), unsafe_allow_html=True)

        # Bench / Starting XI info
        if len(squad_df) >= 11:
            xi_df, captain_id = auto_pick_xi(squad_df)
            captain_name = xi_df.loc[xi_df["player_id"] == captain_id, "player_name"].iloc[0]
            bench = squad_df[~squad_df["player_id"].isin(xi_df["player_id"])]

            xi_pos = xi_df["position"].value_counts()
            formation = f"{xi_pos.get('DEF', 0)}-{xi_pos.get('MID', 0)}-{xi_pos.get('FWD', 0)}"
            total_pts = (xi_df["xgb_predicted_pts"] + xi_df["llm_predicted_pts"]).sum() / 2

            info_html = (
                f'<div class="mc-row" style="margin-top:12px;">'
                f'{metric_card_compact_html("Formation", formation)}'
                f'{metric_card_compact_html("XI Predicted Pts", f"{total_pts:.1f}")}'
                f'{metric_card_compact_html("Captain", captain_name)}'
                f'</div>'
            )
            st.markdown(info_html, unsafe_allow_html=True)

            if not bench.empty:
                bench_names = ", ".join(
                    f"{r['player_name']} ({r['position']})"
                    for _, r in bench.iterrows()
                )
                st.caption(f"Bench: {bench_names}")

    # ======== RIGHT COLUMN: Transfer Planner ========
    with right_col:
        st.markdown(
            '<span class="section-header">Transfer Planner</span>',
            unsafe_allow_html=True,
        )

        # Number of transfers + budget info
        num_transfers = st.radio(
            "Number of transfers",
            options=[1, 2, 3],
            horizontal=True,
            help="Plan 1-3 transfers. Points hit: -4 per extra transfer beyond your free transfer.",
        )

        budget_html = (
            f'<div class="budget-bar">'
            f'<div><span class="budget-label">Squad Value</span><br><span class="budget-value">{total_cost:.1f}m</span></div>'
            f'<div><span class="budget-label">Budget Remaining</span><br><span class="budget-value">{remaining:.1f}m</span></div>'
            f'</div>'
        )
        st.markdown(budget_html, unsafe_allow_html=True)

        # Sort squad by worst performers for the dropdown
        squad_sorted = squad_df.sort_values("combined_value_score", ascending=True)

        for t_idx in range(num_transfers):
            # Transfer section header
            st.markdown(
                f'<div class="tp-section-hdr">Transfer {t_idx + 1}</div>',
                unsafe_allow_html=True,
            )

            # ---- 1. Player Out ----
            st.markdown("**1. Player Out**")
            out_options = {
                f"{r['player_name']} ({r['position']}, {r['team_name']}, {r['price']:.1f}m) — Value: {r['combined_value_score']:.2f}": int(r["player_id"])
                for _, r in squad_sorted.iterrows()
            }

            selected_out = st.selectbox(
                "Select Player",
                options=list(out_options.keys()),
                key=f"transfer_out_{t_idx}",
                label_visibility="collapsed",
            )

            if not selected_out:
                continue

            out_id = out_options[selected_out]
            player_out = squad_df[squad_df["player_id"] == out_id].iloc[0]

            # Out banner with fixtures
            out_team_id = int(player_out["team"])
            out_fixtures = lookahead.get(out_team_id, [])
            fix_badges = fixture_run_html(out_fixtures) if out_fixtures else ""

            st.markdown(
                f'<div class="tp-out-banner">'
                f'<span class="out-pill">OUT</span>'
                f'<span class="out-name">{player_out["player_name"]}</span>'
                f'<span style="color:#888;font-size:0.85em;">Next Fixtures:</span>'
                f'<span class="out-fixtures">{fix_badges}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ---- 2. Recommended Transfers ----
            st.markdown("**2. Recommended Transfers**")

            safe_tab, diff_tab, form_tab, fix_tab = st.tabs(
                ["Safe Picks", "Differentials", "Form", "Fixture"]
            )

            strategies = [
                (safe_tab, "Safe", "High confidence from both models, sorted by value score"),
                (diff_tab, "Differential", "High predicted points but lower confidence -- high risk/reward"),
                (form_tab, "Form", "Players in best recent form with consistent minutes"),
                (fix_tab, "Fixture", "Easiest upcoming fixtures across next 2 gameweeks"),
            ]

            for tab, strategy, description in strategies:
                with tab:
                    st.caption(description)
                    suggestions = get_transfer_suggestions(df, squad_df, player_out, strategy, gw)

                    if suggestions.empty:
                        st.info(f"No {strategy.lower()} picks available for this position and budget.")
                        continue

                    for idx, (_, player_in) in enumerate(suggestions.iterrows()):
                        render_suggestion_card(
                            player_in, player_out, lookahead,
                            idx, f"{t_idx}_{strategy}",
                        )

    # ---- Footer ----
    st.markdown(
        '<div class="fpl-footer">'
        "Built with XGBoost + Llama 3.2 3B (fine-tuned with LoRA on MLX) | "
        "Data from the FPL API | "
        "Predictions are for educational purposes only"
        "</div>",
        unsafe_allow_html=True,
    )


run()
