"""
Predictions Page — Redesigned to match Stitch design mockup.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd

from ui.data_loader import get_enriched_predictions, load_fixture_lookahead
from ui.components import (
    fixture_text,
    fdr_legend_html,
    metric_card_html,
    fixture_badge_html,
    confidence_pill,
    position_badge,
)


def run():
    df = get_enriched_predictions()

    if df is None:
        st.error("No predictions found. Run `python models/predict_next_gw.py` first.")
        return

    gw = int(df["gameweek"].iloc[0])

    # ---- Header Banner ----
    st.markdown(f"""
    <div class="fpl-header">
        <h1>FPL Predictions &amp; Squad Planner</h1>
        <p>Gameweek {gw} Predictions &amp; Transfer Advisor</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Summary Metric Cards ----
    total_players = len(df)
    avg_pts = (df["xgb_predicted_pts"] + df["llm_predicted_pts"]).mean() / 2
    top_player = df.loc[df["combined_value_score"].idxmax()]
    agreement = (abs(df["xgb_predicted_pts"] - df["llm_predicted_pts"]) < 1.0).mean() * 100

    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card_html("Players Analyzed", total_players), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_card_html("Avg Predicted Pts", f"{avg_pts:.1f}"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(
            metric_card_html("Top Pick", f"{top_player['player_name']}", sub=f"{top_player['combined_value_score']:.2f} value"),
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(metric_card_html("Model Agreement", f"{agreement:.1f}%", sub="within 1.0 pts"), unsafe_allow_html=True)

    # ---- Filter Bar ----
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)

    all_positions = ["GK", "DEF", "MID", "FWD"]
    all_teams = sorted(df["team_name"].unique())

    # Row 1: Position pills, Team, Price, Confidence, Fixture
    fc1, fc2, fc3, fc4, fc5 = st.columns([2, 2, 2, 1.5, 1.5])

    with fc1:
        positions = st.multiselect(
            "Position",
            options=all_positions,
            default=all_positions,
        )
    with fc2:
        teams = st.multiselect("Team", options=all_teams, default=[])
    with fc3:
        price_range = st.slider(
            "Price Range",
            min_value=float(df["price"].min()),
            max_value=float(df["price"].max()),
            value=(float(df["price"].min()), float(df["price"].max())),
            step=0.5,
        )
    with fc4:
        min_confidence = st.slider("Min Confidence", 0, 99, 0, step=5)
    with fc5:
        home_away = st.radio("Fixture", ["All", "Home", "Away"], horizontal=True)

    # Sort control
    sort_col1, sort_col2 = st.columns([2, 7])
    with sort_col1:
        sort_by = st.selectbox(
            "Sort by",
            options=[
                "Combined Value Score",
                "XGBoost Predicted Points",
                "LLM Predicted Points",
                "XGBoost Confidence",
                "LLM Confidence",
                "Price (low to high)",
                "Form (3GW)",
            ],
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- FDR Legend Strip ----
    st.markdown(fdr_legend_html(), unsafe_allow_html=True)

    # ---- Apply Filters ----
    filtered = df[df["position"].isin(positions)].copy()

    if teams:
        filtered = filtered[filtered["team_name"].isin(teams)]

    filtered = filtered[
        (filtered["price"] >= price_range[0]) & (filtered["price"] <= price_range[1])
    ]
    filtered = filtered[
        (filtered["xgb_confidence"] >= min_confidence)
        | (filtered["llm_confidence"] >= min_confidence)
    ]

    if home_away == "Home":
        filtered = filtered[filtered["was_home"] == 1]
    elif home_away == "Away":
        filtered = filtered[filtered["was_home"] == 0]

    # Sort
    sort_map = {
        "Combined Value Score": ("combined_value_score", False),
        "XGBoost Predicted Points": ("xgb_predicted_pts", False),
        "LLM Predicted Points": ("llm_predicted_pts", False),
        "XGBoost Confidence": ("xgb_confidence", False),
        "LLM Confidence": ("llm_confidence", False),
        "Price (low to high)": ("price", True),
        "Form (3GW)": ("form_3gw", False),
    }
    sort_col_name, sort_asc = sort_map[sort_by]
    filtered = filtered.sort_values(sort_col_name, ascending=sort_asc).reset_index(drop=True)

    # ---- Build Fixture Text Columns ----
    filtered["GW" + str(gw)] = filtered.apply(
        lambda r: fixture_text(r["opponent"], r["home_away"], r["fixture_difficulty"]), axis=1
    )
    filtered["GW" + str(gw + 1)] = filtered.apply(
        lambda r: fixture_text(r["gw2_opponent"], r["gw2_home_away"], r["gw2_fdr"]), axis=1
    )

    # ---- Hero Table ----
    st.markdown(
        f'<span class="section-header">All Players ({len(filtered)})</span>',
        unsafe_allow_html=True,
    )

    gw_col = f"GW{gw}"
    gw2_col = f"GW{gw + 1}"

    display_df = filtered[
        [
            "player_name",
            "position",
            "team_name",
            "price",
            "last_3_pts",
            gw_col,
            gw2_col,
            "form_3gw",
            "avg_minutes_3gw",
            "xgb_predicted_pts",
            "xgb_confidence",
            "llm_predicted_pts",
            "llm_confidence",
            "combined_value_score",
        ]
    ].copy()

    display_df.columns = [
        "Player",
        "Pos",
        "Team",
        "Price",
        "Last 3 GW",
        gw_col,
        gw2_col,
        "Form",
        "Mins",
        "XGB Pts",
        "XGB Conf",
        "LLM Pts",
        "LLM Conf",
        "Value",
    ]

    # Format confidence as "XX%" strings for cleaner display
    display_df["XGB Conf"] = display_df["XGB Conf"].apply(lambda x: f"{int(x)}%")
    display_df["LLM Conf"] = display_df["LLM Conf"].apply(lambda x: f"{int(x)}%")

    st.dataframe(
        display_df,
        column_config={
            "Price": st.column_config.NumberColumn("Price", format="£%.1f"),
            "Form": st.column_config.NumberColumn("Form", format="%.1f"),
            "Mins": st.column_config.NumberColumn("Mins", format="%d"),
            "XGB Pts": st.column_config.NumberColumn("XGB Pts", format="%.1f"),
            "LLM Pts": st.column_config.NumberColumn("LLM Pts", format="%.1f"),
            "Value": st.column_config.NumberColumn("Value", format="%.2f"),
        },
        use_container_width=True,
        height=700,
        hide_index=True,
    )

    # ---- Player Deep Dive ----
    with st.expander("Player Deep Dive", expanded=False):
        player_names = sorted(filtered["player_name"].unique())
        selected_player = st.selectbox("Select a player", player_names, key="deep_dive_player")

        if selected_player:
            p = filtered[filtered["player_name"] == selected_player].iloc[0]

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("#### XGBoost Prediction")
                st.metric("Predicted Points", f"{p['xgb_predicted_pts']:.1f}")
                st.metric("Confidence", f"{p['xgb_confidence']}%")
                st.metric("Value Score", f"{p['xgb_value_score']:.2f}")

            with col_right:
                st.markdown("#### Fine-Tuned LLM Prediction")
                st.metric("Predicted Points", f"{p['llm_predicted_pts']:.1f}")
                st.metric("Confidence", f"{p['llm_confidence']}%")
                st.metric("Value Score", f"{p['llm_value_score']:.2f}")

            st.markdown("#### Player Context")
            ctx1, ctx2, ctx3, ctx4 = st.columns(4)
            with ctx1:
                ha = "Home" if p["was_home"] == 1 else "Away"
                st.markdown(f"**Fixture:** {ha} vs {p['opponent']}")
                st.markdown(f"**FDR:** {int(p['fixture_difficulty'])}")
            with ctx2:
                st.markdown(f"**Price:** {p['price']:.1f}m")
                st.markdown(f"**Position:** {p['position']}")
            with ctx3:
                st.markdown(f"**Form (3GW):** {p['form_3gw']:.1f}")
                st.markdown(f"**ICT (3GW):** {p['ict_index_3gw']:.1f}")
            with ctx4:
                st.markdown(f"**Avg Minutes (3GW):** {p['avg_minutes_3gw']:.0f}")
                gw2_fix = f"{p['gw2_opponent']}({p['gw2_home_away']})" if p["gw2_opponent"] != "-" else "-"
                st.markdown(f"**Next Fixture:** {gw2_fix}")

    # ---- Model Agreement / Disagreement ----
    with st.expander("Model Agreement & Disagreement", expanded=False):
        filtered_copy = filtered.copy()
        filtered_copy["pts_diff"] = abs(
            filtered_copy["xgb_predicted_pts"] - filtered_copy["llm_predicted_pts"]
        )

        agree_tab, disagree_tab = st.tabs(["Most Agreement", "Most Disagreement"])

        with agree_tab:
            st.caption("Players where both models predict similar points (high conviction picks)")
            agreed = filtered_copy.nsmallest(15, "pts_diff")[
                [
                    "player_name", "position", "team_name", "opponent", "home_away",
                    "xgb_predicted_pts", "llm_predicted_pts", "pts_diff", "combined_value_score",
                ]
            ].copy()
            agreed.columns = [
                "Player", "Pos", "Team", "Opp", "H/A",
                "XGB Pts", "LLM Pts", "Diff", "Value",
            ]
            st.dataframe(
                agreed.style.format(
                    {"XGB Pts": "{:.1f}", "LLM Pts": "{:.1f}", "Diff": "{:.1f}", "Value": "{:.2f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        with disagree_tab:
            st.caption("Players where the models disagree most (proceed with caution)")
            disagreed = filtered_copy.nlargest(15, "pts_diff")[
                [
                    "player_name", "position", "team_name", "opponent", "home_away",
                    "xgb_predicted_pts", "llm_predicted_pts", "pts_diff", "combined_value_score",
                ]
            ].copy()
            disagreed.columns = [
                "Player", "Pos", "Team", "Opp", "H/A",
                "XGB Pts", "LLM Pts", "Diff", "Value",
            ]
            st.dataframe(
                disagreed.style.format(
                    {"XGB Pts": "{:.1f}", "LLM Pts": "{:.1f}", "Diff": "{:.1f}", "Value": "{:.2f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

    # ---- Footer ----
    st.markdown(
        '<div class="fpl-footer">'
        'Built with XGBoost + Llama 3.2 3B (fine-tuned with LoRA on MLX) | '
        'Data from the FPL API | '
        'Predictions are for educational purposes only'
        '</div>',
        unsafe_allow_html=True,
    )


run()
