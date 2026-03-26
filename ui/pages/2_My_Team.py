"""
My Team Page — Squad builder + multi-transfer planner.
"""

import streamlit as st
import pandas as pd

from ui.data_loader import get_enriched_predictions, load_fixture_lookahead
from ui.components import (
    fixture_badge_html,
    fixture_run_html,
    player_card_html,
    metric_card_html,
    delta_html,
    fdr_legend_html,
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


def render_pitch(squad_df, gw, lookahead):
    """Render squad in a pitch layout."""
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pos_players = squad_df[squad_df["position"] == pos].sort_values(
            "combined_value_score", ascending=False
        )

        if len(pos_players) == 0:
            continue

        cols = st.columns(max(len(pos_players), 1))
        for i, (_, player) in enumerate(pos_players.iterrows()):
            team_id = int(player["team"])
            fixtures = lookahead.get(team_id, [])

            with cols[i]:
                st.markdown(
                    player_card_html(
                        player["player_name"],
                        player["team_name"],
                        player["price"],
                        player["combined_value_score"],
                        fixtures,
                    ),
                    unsafe_allow_html=True,
                )


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


def render_suggestion_row(player_in, player_out, gw, lookahead, idx, transfer_idx):
    """Render a single transfer suggestion with apply button."""
    team_id_in = int(player_in["team"])
    fixtures_in = lookahead.get(team_id_in, [])

    pts_delta = player_in["combined_value_score"] - player_out["combined_value_score"]
    xgb_delta = player_in["xgb_predicted_pts"] - player_out["xgb_predicted_pts"]
    llm_delta = player_in["llm_predicted_pts"] - player_out["llm_predicted_pts"]
    cost_delta = player_in["price"] - player_out["price"]

    c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 1.5])

    with c1:
        fixtures_html = fixture_run_html(fixtures_in) if fixtures_in else ""
        st.markdown(
            f"**{player_in['player_name']}** "
            f"<span style='color:#888;font-size:0.85em;'>"
            f"{player_in['team_name']} &middot; {player_in['price']:.1f}m</span>"
            f"<br>{fixtures_html}",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"XGB: **{player_in['xgb_predicted_pts']:.1f}** pts "
            f"({'+' if xgb_delta >= 0 else ''}{xgb_delta:.1f})<br>"
            f"LLM: **{player_in['llm_predicted_pts']:.1f}** pts "
            f"({'+' if llm_delta >= 0 else ''}{llm_delta:.1f})",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(f"Value: **{player_in['combined_value_score']:.2f}**")
        st.markdown(delta_html(pts_delta, "value"), unsafe_allow_html=True)

    with c4:
        cost_str = f"+{cost_delta:.1f}m" if cost_delta >= 0 else f"{cost_delta:.1f}m"
        st.markdown(f"Cost: **{cost_str}**")

    with c5:
        if st.button("Apply", key=f"apply_{transfer_idx}_{idx}", type="primary"):
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
        <h1>My Team</h1>
        <p>Squad Builder & Transfer Advisor &mdash; GW {gw}</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Squad Builder ----
    st.markdown('<span class="section-header">Squad Builder</span>', unsafe_allow_html=True)

    # Build player options for multiselect
    player_options = {}
    for _, row in df.iterrows():
        label = f"{row['player_name']} - {row['position']} - {row['team_name']} - {row['price']:.1f}m"
        player_options[label] = int(row["player_id"])

    # Reverse mapping for defaults
    id_to_label = {v: k for k, v in player_options.items()}
    current_labels = [id_to_label[pid] for pid in st.session_state.squad if pid in id_to_label]

    selected_labels = st.multiselect(
        "Select your 15 players",
        options=sorted(player_options.keys()),
        default=current_labels,
        max_selections=15,
        help="Search by player name. Select up to 15 players (2 GK, 5 DEF, 5 MID, 3 FWD).",
    )

    # Update session state
    new_squad = [player_options[label] for label in selected_labels]
    if new_squad != st.session_state.squad:
        st.session_state.squad = new_squad

    squad_df = get_squad_df(df)

    if squad_df.empty:
        st.info("Select players above to build your squad.")
        return

    # ---- Validation ----
    warnings = validate_squad(df, st.session_state.squad)
    for w in warnings:
        st.warning(w)

    # ---- Squad Composition & Budget ----
    total_cost = squad_df["price"].sum()
    remaining = TOTAL_BUDGET - total_cost
    pos_counts = squad_df["position"].value_counts().to_dict()

    comp_cols = st.columns(6)
    with comp_cols[0]:
        st.markdown(
            metric_card_html("Budget Left", f"{remaining:.1f}m", sub=f"{total_cost:.1f}m spent"),
            unsafe_allow_html=True,
        )
    with comp_cols[1]:
        st.markdown(
            metric_card_html("Players", f"{len(squad_df)}/15"),
            unsafe_allow_html=True,
        )
    for i, pos in enumerate(["GK", "DEF", "MID", "FWD"]):
        with comp_cols[i + 2]:
            count = pos_counts.get(pos, 0)
            limit = POSITION_LIMITS[pos]
            st.markdown(
                metric_card_html(pos, f"{count}/{limit}"),
                unsafe_allow_html=True,
            )

    # ---- FDR Legend ----
    st.markdown(fdr_legend_html(), unsafe_allow_html=True)

    # ---- Pitch Layout ----
    st.markdown('<span class="section-header">Your Squad</span>', unsafe_allow_html=True)
    render_pitch(squad_df, gw, lookahead)

    # ---- Squad Overview Table ----
    if len(squad_df) >= 11:
        st.markdown('<span class="section-header">Starting XI</span>', unsafe_allow_html=True)

        xi_df, captain_id = auto_pick_xi(squad_df)

        captain_name = xi_df.loc[xi_df["player_id"] == captain_id, "player_name"].iloc[0]
        bench = squad_df[~squad_df["player_id"].isin(xi_df["player_id"])]

        # Formation
        xi_pos = xi_df["position"].value_counts()
        formation = f"{xi_pos.get('DEF', 0)}-{xi_pos.get('MID', 0)}-{xi_pos.get('FWD', 0)}"

        info_cols = st.columns(3)
        with info_cols[0]:
            st.markdown(metric_card_html("Formation", formation), unsafe_allow_html=True)
        with info_cols[1]:
            total_pts = (xi_df["xgb_predicted_pts"] + xi_df["llm_predicted_pts"]).sum() / 2
            st.markdown(
                metric_card_html("XI Predicted Pts", f"{total_pts:.1f}"),
                unsafe_allow_html=True,
            )
        with info_cols[2]:
            st.markdown(
                metric_card_html("Captain", captain_name, sub="highest value score"),
                unsafe_allow_html=True,
            )

        # XI table
        xi_display = xi_df[
            ["player_name", "position", "team_name", "price", "opponent", "home_away",
             "fixture_difficulty", "gw2_opponent", "gw2_home_away",
             "xgb_predicted_pts", "xgb_confidence", "llm_predicted_pts",
             "llm_confidence", "combined_value_score"]
        ].copy()
        xi_display.columns = [
            "Player", "Pos", "Team", "Price", "Opp", "H/A", "FDR",
            "Next Opp", "Next H/A",
            "XGB Pts", "XGB Conf", "LLM Pts", "LLM Conf", "Value",
        ]
        st.dataframe(
            xi_display.style.format({
                "Price": "{:.1f}", "XGB Pts": "{:.1f}", "LLM Pts": "{:.1f}", "Value": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        if not bench.empty:
            st.markdown("**Bench:**")
            bench_display = bench[
                ["player_name", "position", "team_name", "price",
                 "xgb_predicted_pts", "llm_predicted_pts", "combined_value_score"]
            ].copy()
            bench_display.columns = ["Player", "Pos", "Team", "Price", "XGB Pts", "LLM Pts", "Value"]
            st.dataframe(
                bench_display.style.format({
                    "Price": "{:.1f}", "XGB Pts": "{:.1f}", "LLM Pts": "{:.1f}", "Value": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    # ---- Transfer Planner ----
    st.markdown("---")
    st.markdown('<span class="section-header">Transfer Planner</span>', unsafe_allow_html=True)

    num_transfers = st.radio(
        "Number of transfers",
        options=[1, 2, 3],
        horizontal=True,
        help="Plan 1-3 transfers. Points hit: -4 per extra transfer beyond your free transfer.",
    )

    # Budget tracker
    total_cost = squad_df["price"].sum()
    st.markdown(
        f"""<div class="budget-bar">
        <div><span class="budget-label">Squad Value</span><br><span class="budget-value">{total_cost:.1f}m</span></div>
        <div><span class="budget-label">Budget Remaining</span><br><span class="budget-value">{TOTAL_BUDGET - total_cost:.1f}m</span></div>
        <div><span class="budget-label">Transfers</span><br><span class="budget-value">{num_transfers}</span></div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Sort squad by worst performers
    squad_sorted = squad_df.sort_values("combined_value_score", ascending=True)

    for t_idx in range(num_transfers):
        st.markdown(f"#### Transfer {t_idx + 1}")

        # Player out selection
        out_options = {
            f"{r['player_name']} ({r['position']}, {r['team_name']}, {r['price']:.1f}m) — Value: {r['combined_value_score']:.2f}": int(r["player_id"])
            for _, r in squad_sorted.iterrows()
        }

        selected_out = st.selectbox(
            "Player Out",
            options=list(out_options.keys()),
            key=f"transfer_out_{t_idx}",
        )

        if not selected_out:
            continue

        out_id = out_options[selected_out]
        player_out = squad_df[squad_df["player_id"] == out_id].iloc[0]

        # Show outgoing player fixture info
        out_team_id = int(player_out["team"])
        out_fixtures = lookahead.get(out_team_id, [])
        if out_fixtures:
            st.markdown(
                f"<span class='transfer-out'>OUT:</span> {player_out['player_name']} "
                f"&mdash; Next fixtures: {fixture_run_html(out_fixtures)}",
                unsafe_allow_html=True,
            )

        # Strategy tabs
        safe_tab, diff_tab, form_tab, fix_tab = st.tabs(
            ["Safe Picks", "Differential", "Form", "Fixture"]
        )

        strategies = [
            (safe_tab, "Safe", "High confidence from both models, sorted by value score"),
            (diff_tab, "Differential", "High predicted points but lower confidence — high risk/reward"),
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
                    render_suggestion_row(player_in, player_out, gw, lookahead, idx, f"{t_idx}_{strategy}")
                    if idx < len(suggestions) - 1:
                        st.markdown(
                            "<hr style='margin:4px 0;border:none;border-top:1px solid #f0f0f0;'>",
                            unsafe_allow_html=True,
                        )

    # Footer
    st.markdown("---")
    st.caption(
        "Built with XGBoost + Llama 3.2 3B (fine-tuned with LoRA on MLX) | "
        "Data from the FPL API | "
        "Predictions are for educational purposes only"
    )


run()
