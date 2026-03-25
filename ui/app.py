"""
FPL Points Predictor — Side-by-Side Comparison Dashboard

Run with: streamlit run ui/app.py
"""

import os
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_PATH = os.path.join(ROOT, "data", "processed", "next_gw_predictions.csv")
XGB_RESULTS_PATH = os.path.join(ROOT, "eval", "xgboost_results.json")
LLM_RESULTS_PATH = os.path.join(ROOT, "eval", "llm_results_summary.json")

st.set_page_config(
    page_title="FPL Points Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px 0 10px;
    }
    .main-header h1 {
        color: #0f3460;
        font-size: 2.2em;
        margin-bottom: 4px;
    }
    .main-header p {
        color: #888;
        font-size: 1.1em;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-card h3 {
        color: #888;
        font-size: 0.8em;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-size: 1.8em;
        font-weight: 800;
        color: #0f3460;
    }
    .conf-high { color: #2e7d32; font-weight: 700; }
    .conf-med { color: #e65100; font-weight: 700; }
    .conf-low { color: #c62828; font-weight: 700; }
    .badge-home {
        background: #e8f5e9; color: #2e7d32;
        padding: 2px 8px; border-radius: 8px; font-weight: 600; font-size: 0.85em;
    }
    .badge-away {
        background: #fce4ec; color: #c62828;
        padding: 2px 8px; border-radius: 8px; font-weight: 600; font-size: 0.85em;
    }
    div[data-testid="stDataFrame"] table {
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


def conf_color(val):
    if val >= 75:
        return "conf-high"
    elif val >= 50:
        return "conf-med"
    return "conf-low"


def load_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        return None
    return pd.read_csv(PREDICTIONS_PATH)


def main():
    df = load_predictions()

    if df is None:
        st.error("No predictions found. Run `python models/predict_next_gw.py` first.")
        return

    gw = int(df["gameweek"].iloc[0])

    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>FPL Points Predictor</h1>
        <p>Gameweek {gw} predictions &mdash; XGBoost vs Fine-Tuned LLM</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar filters
    st.sidebar.header("Filters")

    positions = st.sidebar.multiselect(
        "Position",
        options=["GK", "DEF", "MID", "FWD"],
        default=["GK", "DEF", "MID", "FWD"],
    )

    max_price = st.sidebar.slider(
        "Max Price",
        min_value=float(df["price"].min()),
        max_value=float(df["price"].max()),
        value=float(df["price"].max()),
        step=0.5,
    )

    min_confidence = st.sidebar.slider(
        "Min Confidence (either model)",
        min_value=0,
        max_value=99,
        value=0,
        step=5,
    )

    home_away = st.sidebar.radio(
        "Fixture",
        options=["All", "Home only", "Away only"],
    )

    sort_by = st.sidebar.selectbox(
        "Sort by",
        options=[
            "Combined Value Score",
            "XGBoost Predicted Points",
            "LLM Predicted Points",
            "XGBoost Confidence",
            "LLM Confidence",
            "Price (low to high)",
        ],
    )

    # Apply filters
    filtered = df[df["position"].isin(positions)]
    filtered = filtered[filtered["price"] <= max_price]
    filtered = filtered[
        (filtered["xgb_confidence"] >= min_confidence) |
        (filtered["llm_confidence"] >= min_confidence)
    ]
    if home_away == "Home only":
        filtered = filtered[filtered["was_home"] == 1]
    elif home_away == "Away only":
        filtered = filtered[filtered["was_home"] == 0]

    # Sort
    sort_map = {
        "Combined Value Score": ("combined_value_score", False),
        "XGBoost Predicted Points": ("xgb_predicted_pts", False),
        "LLM Predicted Points": ("llm_predicted_pts", False),
        "XGBoost Confidence": ("xgb_confidence", False),
        "LLM Confidence": ("llm_confidence", False),
        "Price (low to high)": ("price", True),
    }
    sort_col, sort_asc = sort_map[sort_by]
    filtered = filtered.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    # ---- TOP 30 VALUE PICKS ----
    st.markdown("---")
    st.subheader(f"Top 30 Value Picks — GW {gw}")
    st.caption("Ranked by combined value score (predicted points x confidence, averaged across both models)")

    top30 = filtered.head(30)

    # Display as styled table
    for i, (_, row) in enumerate(top30.iterrows()):
        rank = i + 1
        home_badge = "badge-home" if row["was_home"] == 1 else "badge-away"
        ha_label = "HOME" if row["was_home"] == 1 else "AWAY"
        fdr_stars = int(row["fixture_difficulty"])

        col1, col2, col3, col4, col5, col6 = st.columns([0.4, 2.2, 1.5, 2, 2, 1.2])

        with col1:
            st.markdown(f"**#{rank}**")

        with col2:
            st.markdown(
                f"**{row['player_name']}** &nbsp; "
                f"<span style='color:#888; font-size:0.85em;'>{row['position']} &middot; {row['team_name']} &middot; {row['price']:.1f}m</span>",
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"<span class='{home_badge}'>{ha_label}</span> vs {row['opponent']} "
                f"<span style='color:#888; font-size:0.82em;'>(FDR {'*' * fdr_stars})</span>",
                unsafe_allow_html=True,
            )

        with col4:
            xgb_cc = conf_color(row["xgb_confidence"])
            st.markdown(
                f"XGB: **{row['xgb_predicted_pts']:.1f} pts** "
                f"<span class='{xgb_cc}'>{row['xgb_confidence']}%</span>",
                unsafe_allow_html=True,
            )

        with col5:
            llm_cc = conf_color(row["llm_confidence"])
            st.markdown(
                f"LLM: **{row['llm_predicted_pts']:.1f} pts** "
                f"<span class='{llm_cc}'>{row['llm_confidence']}%</span>",
                unsafe_allow_html=True,
            )

        with col6:
            st.markdown(f"**{row['combined_value_score']:.2f}**")

        if rank < 30 and rank < len(top30):
            st.markdown(
                "<hr style='margin: 2px 0; border: none; border-top: 1px solid #f0f0f0;'>",
                unsafe_allow_html=True,
            )

    # ---- FULL TABLE ----
    st.markdown("---")
    st.subheader(f"All Players ({len(filtered)} shown)")

    display_df = filtered[[
        "player_name", "position", "team_name", "price",
        "opponent", "home_away", "fixture_difficulty",
        "form_3gw", "ict_index_3gw",
        "xgb_predicted_pts", "xgb_confidence",
        "llm_predicted_pts", "llm_confidence",
        "combined_value_score",
    ]].copy()

    display_df.columns = [
        "Player", "Pos", "Team", "Price",
        "Opponent", "H/A", "FDR",
        "Form (3GW)", "ICT (3GW)",
        "XGB Pts", "XGB Conf%",
        "LLM Pts", "LLM Conf%",
        "Value Score",
    ]

    st.dataframe(
        display_df.style
        .format({
            "Price": "{:.1f}",
            "Form (3GW)": "{:.1f}",
            "ICT (3GW)": "{:.1f}",
            "XGB Pts": "{:.1f}",
            "LLM Pts": "{:.1f}",
            "Value Score": "{:.2f}",
        })
        .background_gradient(subset=["XGB Pts", "LLM Pts"], cmap="Greens", vmin=0, vmax=8)
        .background_gradient(subset=["XGB Conf%", "LLM Conf%"], cmap="Blues", vmin=0, vmax=100)
        .background_gradient(subset=["Value Score"], cmap="YlOrRd", vmin=0),
        use_container_width=True,
        height=600,
    )

    # ---- SIDE-BY-SIDE PLAYER LOOKUP ----
    st.markdown("---")
    st.subheader("Player Deep Dive")

    player_names = sorted(filtered["player_name"].unique())
    selected_player = st.selectbox("Select a player", player_names)

    if selected_player:
        p = filtered[filtered["player_name"] == selected_player].iloc[0]

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### XGBoost Prediction")
            st.metric("Predicted Points", f"{p['xgb_predicted_pts']:.1f}")
            st.metric("Confidence", f"{p['xgb_confidence']}%")
            st.metric("Value Score", f"{p['xgb_value_score']:.2f}")

        with col_right:
            st.markdown("### Fine-Tuned LLM Prediction")
            st.metric("Predicted Points", f"{p['llm_predicted_pts']:.1f}")
            st.metric("Confidence", f"{p['llm_confidence']}%")
            st.metric("Value Score", f"{p['llm_value_score']:.2f}")

        st.markdown("#### Player Context")
        ctx_col1, ctx_col2, ctx_col3 = st.columns(3)
        with ctx_col1:
            ha = "Home" if p["was_home"] == 1 else "Away"
            st.markdown(f"**Fixture:** {ha} vs {p['opponent']} (FDR {int(p['fixture_difficulty'])})")
            st.markdown(f"**Price:** {p['price']:.1f}m")
        with ctx_col2:
            st.markdown(f"**Form (3GW):** {p['form_3gw']:.1f}")
            st.markdown(f"**ICT (3GW):** {p['ict_index_3gw']:.1f}")
        with ctx_col3:
            st.markdown(f"**Avg Minutes (3GW):** {p['avg_minutes_3gw']:.0f}")
            st.markdown(f"**Position:** {p['position']}")

    # ---- MODEL AGREEMENT ----
    st.markdown("---")
    st.subheader("Where Do The Models Agree / Disagree?")

    filtered_copy = filtered.copy()
    filtered_copy["pts_diff"] = abs(
        filtered_copy["xgb_predicted_pts"] - filtered_copy["llm_predicted_pts"]
    )

    agree_tab, disagree_tab = st.tabs(["Most Agreement", "Most Disagreement"])

    with agree_tab:
        st.caption("Players where both models predict similar points (high conviction picks)")
        agreed = filtered_copy.nsmallest(15, "pts_diff")[
            ["player_name", "position", "team_name", "opponent", "home_away",
             "xgb_predicted_pts", "llm_predicted_pts", "pts_diff", "combined_value_score"]
        ].copy()
        agreed.columns = ["Player", "Pos", "Team", "Opp", "H/A",
                          "XGB Pts", "LLM Pts", "Difference", "Value"]
        st.dataframe(agreed.style.format({
            "XGB Pts": "{:.1f}", "LLM Pts": "{:.1f}",
            "Difference": "{:.1f}", "Value": "{:.2f}",
        }), use_container_width=True)

    with disagree_tab:
        st.caption("Players where the models disagree most (proceed with caution)")
        disagreed = filtered_copy.nlargest(15, "pts_diff")[
            ["player_name", "position", "team_name", "opponent", "home_away",
             "xgb_predicted_pts", "llm_predicted_pts", "pts_diff", "combined_value_score"]
        ].copy()
        disagreed.columns = ["Player", "Pos", "Team", "Opp", "H/A",
                             "XGB Pts", "LLM Pts", "Difference", "Value"]
        st.dataframe(disagreed.style.format({
            "XGB Pts": "{:.1f}", "LLM Pts": "{:.1f}",
            "Difference": "{:.1f}", "Value": "{:.2f}",
        }), use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption(
        "Built with XGBoost + Llama 3.2 3B (fine-tuned with LoRA on MLX) | "
        "Data from the FPL API | "
        "Predictions are for educational purposes only"
    )


if __name__ == "__main__":
    main()
