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

    # ---- Check if Claude predictions exist ----
    has_claude = "claude_predicted_pts" in df.columns and df["claude_predicted_pts"].notna().any()

    # Metric cards placeholder — rendered after model filter is known
    metric_cards_spot = st.container()

    # ---- Filter Bar ----
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

    # Model filter and Sort control
    mf_col, sort_col1, sort_col2 = st.columns([2, 2, 5])
    with mf_col:
        all_models = ["XGBoost", "LLM"]
        if has_claude:
            all_models.append("Claude")
        selected_models = st.multiselect("Models", options=all_models, default=all_models)
    with sort_col1:
        sort_options = ["Combined Value Score", "Price (low to high)", "Form (3GW)"]
        if "XGBoost" in selected_models:
            sort_options.insert(1, "XGBoost Predicted Points")
            sort_options.append("XGBoost Confidence")
        if "LLM" in selected_models:
            idx = sort_options.index("Price (low to high)")
            sort_options.insert(idx, "LLM Predicted Points")
            sort_options.append("LLM Confidence")
        if "Claude" in selected_models:
            idx = sort_options.index("Price (low to high)")
            sort_options.insert(idx, "Claude Predicted Points")
            sort_options.append("Claude Confidence")
        sort_by = st.selectbox("Sort by", options=sort_options)

    # ---- Recalculate combined value score based on selected models ----
    MODEL_MAP = {
        "XGBoost": ("xgb_value_score", "xgb_predicted_pts", "xgb_confidence"),
        "LLM": ("llm_value_score", "llm_predicted_pts", "llm_confidence"),
        "Claude": ("claude_value_score", "claude_predicted_pts", "claude_confidence"),
    }

    if selected_models:
        value_cols = [MODEL_MAP[m][0] for m in selected_models if MODEL_MAP[m][0] in df.columns]
        if value_cols:
            df["combined_value_score"] = df[value_cols].mean(axis=1).round(2)

    # ---- Render Metric Cards (into the placeholder above filters) ----
    with metric_cards_spot:
        pts_cols = [MODEL_MAP[m][1] for m in selected_models if MODEL_MAP[m][1] in df.columns]
        avg_pts = df[pts_cols].mean(axis=1).mean() if pts_cols else 0
        top_player = df.loc[df["combined_value_score"].idxmax()]

        # Agreement: only if 2+ models selected
        if len(pts_cols) >= 2:
            from itertools import combinations
            diffs = [abs(df[a] - df[b]) for a, b in combinations(pts_cols, 2)]
            agreement = (sum(d < 1.0 for d in diffs) / len(diffs)).mean() * 100
            agreement_sub = "within 1.0 pts"
        else:
            agreement = 100.0
            agreement_sub = "single model"

        cols = st.columns(4)
        with cols[0]:
            st.markdown(metric_card_html("Players Analyzed", len(df)), unsafe_allow_html=True)
        with cols[1]:
            st.markdown(metric_card_html("Avg Predicted Pts", f"{avg_pts:.1f}"), unsafe_allow_html=True)
        with cols[2]:
            st.markdown(
                metric_card_html("Top Pick", f"{top_player['player_name']}", sub=f"{top_player['combined_value_score']:.2f} value"),
                unsafe_allow_html=True,
            )
        with cols[3]:
            st.markdown(metric_card_html("Model Agreement", f"{agreement:.1f}%", sub=agreement_sub), unsafe_allow_html=True)

    # ---- FDR Legend Strip ----
    st.markdown(fdr_legend_html(), unsafe_allow_html=True)

    # ---- Apply Filters ----
    filtered = df[df["position"].isin(positions)].copy()

    if teams:
        filtered = filtered[filtered["team_name"].isin(teams)]

    filtered = filtered[
        (filtered["price"] >= price_range[0]) & (filtered["price"] <= price_range[1])
    ]

    # Confidence filter: pass if ANY selected model meets threshold
    if selected_models:
        conf_cols = [MODEL_MAP[m][2] for m in selected_models if MODEL_MAP[m][2] in filtered.columns]
        if conf_cols:
            conf_mask = filtered[conf_cols].ge(min_confidence).any(axis=1)
            filtered = filtered[conf_mask]

    if home_away == "Home":
        filtered = filtered[filtered["was_home"] == 1]
    elif home_away == "Away":
        filtered = filtered[filtered["was_home"] == 0]

    # Sort
    sort_map = {
        "Combined Value Score": ("combined_value_score", False),
        "XGBoost Predicted Points": ("xgb_predicted_pts", False),
        "LLM Predicted Points": ("llm_predicted_pts", False),
        "Claude Predicted Points": ("claude_predicted_pts", False),
        "XGBoost Confidence": ("xgb_confidence", False),
        "LLM Confidence": ("llm_confidence", False),
        "Claude Confidence": ("claude_confidence", False),
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

    # Build table columns dynamically based on selected models
    table_cols = [
        "photo_url", "player_name", "position", "team_name", "price",
        "last_3_pts", gw_col, gw2_col, "form_3gw", "avg_minutes_3gw",
    ]
    col_names = [
        "Photo", "Player", "Pos", "Team", "Price",
        "Last 3 GW", gw_col, gw2_col, "Form", "Mins",
    ]

    if "XGBoost" in selected_models:
        table_cols.extend(["xgb_predicted_pts", "xgb_confidence"])
        col_names.extend(["XGB Pts", "XGB Conf"])
    if "LLM" in selected_models:
        table_cols.extend(["llm_predicted_pts", "llm_confidence"])
        col_names.extend(["LLM Pts", "LLM Conf"])
    if "Claude" in selected_models and has_claude:
        table_cols.extend(["claude_predicted_pts", "claude_confidence"])
        col_names.extend(["Claude Pts", "Claude Conf"])

    table_cols.append("combined_value_score")
    col_names.append("Value")

    display_df = filtered[table_cols].copy()
    display_df.columns = col_names

    # Format confidence as "XX%" strings
    if "XGB Conf" in display_df.columns:
        display_df["XGB Conf"] = display_df["XGB Conf"].apply(lambda x: f"{int(x)}%")
    if "LLM Conf" in display_df.columns:
        display_df["LLM Conf"] = display_df["LLM Conf"].apply(lambda x: f"{int(x)}%")
    if "Claude Conf" in display_df.columns:
        display_df["Claude Conf"] = display_df["Claude Conf"].apply(lambda x: f"{int(x)}%")

    col_config = {
        "Photo": st.column_config.ImageColumn("", width="small"),
        "Price": st.column_config.NumberColumn("Price", format="£%.1f"),
        "Form": st.column_config.NumberColumn("Form", format="%.1f"),
        "Mins": st.column_config.NumberColumn("Mins", format="%d"),
        "Value": st.column_config.NumberColumn("Value", format="%.2f"),
    }
    if "XGB Pts" in display_df.columns:
        col_config["XGB Pts"] = st.column_config.NumberColumn("XGB Pts", format="%.1f")
    if "LLM Pts" in display_df.columns:
        col_config["LLM Pts"] = st.column_config.NumberColumn("LLM Pts", format="%.1f")
    if "Claude Pts" in display_df.columns:
        col_config["Claude Pts"] = st.column_config.NumberColumn("Claude Pts", format="%.1f")

    st.dataframe(
        display_df,
        column_config=col_config,
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

            model_cols = st.columns(3 if has_claude else 2)

            with model_cols[0]:
                st.markdown("#### XGBoost")
                st.metric("Predicted Points", f"{p['xgb_predicted_pts']:.1f}")
                st.metric("Confidence", f"{p['xgb_confidence']}%")
                st.metric("Value Score", f"{p['xgb_value_score']:.2f}")

            with model_cols[1]:
                st.markdown("#### Fine-Tuned LLM")
                st.metric("Predicted Points", f"{p['llm_predicted_pts']:.1f}")
                st.metric("Confidence", f"{p['llm_confidence']}%")
                st.metric("Value Score", f"{p['llm_value_score']:.2f}")

            if has_claude:
                with model_cols[2]:
                    st.markdown("#### Claude (Haiku)")
                    st.metric("Predicted Points", f"{p['claude_predicted_pts']:.1f}")
                    st.metric("Confidence", f"{int(p['claude_confidence'])}%")
                    st.metric("Value Score", f"{p['claude_value_score']:.2f}")
                    if pd.notna(p.get("claude_reasoning")):
                        st.caption(f"_{p['claude_reasoning']}_")

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
        'Built with XGBoost + Llama 3.2 3B (fine-tuned with LoRA on MLX) + Claude Haiku | '
        'Data from the FPL API | '
        'Predictions are for educational purposes only'
        '</div>',
        unsafe_allow_html=True,
    )


run()
