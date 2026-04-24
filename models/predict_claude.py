"""
Use Claude (Anthropic API) as a third prediction model.

Sends player stats to Claude Haiku for point predictions with confidence
and reasoning. Optionally uses Claude Sonnet for transfer analysis.

Requires ANTHROPIC_API_KEY environment variable.

Usage:
    python models/predict_claude.py                # predictions only
    python models/predict_claude.py --transfers     # + transfer analysis
"""

import argparse
import json
import os
import re
import time
import numpy as np
import pandas as pd
from anthropic import Anthropic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(ROOT, "data", "processed", "features.csv")
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUTPUT_PATH = os.path.join(ROOT, "data", "processed", "claude_predictions.csv")
PREDICTIONS_PATH = os.path.join(ROOT, "data", "processed", "next_gw_predictions.csv")

PREDICTION_MODEL = "claude-haiku-4-5-20251001"
TRANSFER_MODEL = "claude-sonnet-4-6-20250514"

# System prompt for predictions
PREDICTION_SYSTEM = """You are an expert Fantasy Premier League analyst. Given a player's recent stats and upcoming fixture, predict their FPL points for the next gameweek.

You must respond in EXACTLY this JSON format:
{"predicted_pts": 4.2, "confidence": 72, "reasoning": "Brief 1-sentence explanation"}

Rules:
- predicted_pts: a decimal number (e.g. 2.0, 5.5, 8.3). Consider that the average FPL player scores ~3 points per gameweek. 2 points is a typical blank, 5-7 is a good return, 10+ is a haul.
- confidence: integer 0-100. Higher when the prediction is more certain (nailed starter, clear fixture, consistent form). Lower for volatile players or tricky fixtures.
- reasoning: ONE sentence max explaining your prediction.

Use your knowledge of football, FPL scoring, and the stats provided. Consider:
- Form trends (is the player improving or declining?)
- Fixture difficulty (FDR 1-2 is easy, 4-5 is hard)
- Position (GKs/DEFs get clean sheet points, MIDs/FWDs get attacking points)
- Home advantage (home players score more on average)
- ICT index (higher = more involved in attacks)
- Price (expensive players are expected to return more)"""

# Few-shot examples for better calibration
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "Player: Salah (MID)\nGameweek: 15\nFixture: Home (FDR: 2)\nForm (last 3 GW avg): 8.3 pts\nForm (last 5 GW avg): 7.2 pts\nMinutes (last 3 GW avg): 90.0\nGoals per 90 (season): 0.62\nAssists per 90 (season): 0.31\nICT index (last 3 GW avg): 12.1\nClean sheet % (season): 0.0%\nBonus avg (last 3 GW): 1.7\nTeam goals scored (last 3 GW): 9\nTeam goals conceded (last 3 GW): 2\nOpponent goals conceded (season avg per GW): 1.8\nPrice: 13.2"
    },
    {
        "role": "assistant",
        "content": '{"predicted_pts": 7.5, "confidence": 78, "reasoning": "Elite form of 8.3 avg plus easy home fixture (FDR 2) against a leaky defence makes a big return likely."}'
    },
    {
        "role": "user",
        "content": "Player: Mykolenko (DEF)\nGameweek: 18\nFixture: Away (FDR: 5)\nForm (last 3 GW avg): 1.3 pts\nForm (last 5 GW avg): 1.5 pts\nMinutes (last 3 GW avg): 75.0\nGoals per 90 (season): 0.00\nAssists per 90 (season): 0.02\nICT index (last 3 GW avg): 1.2\nClean sheet % (season): 15.0%\nBonus avg (last 3 GW): 0.0\nTeam goals scored (last 3 GW): 2\nTeam goals conceded (last 3 GW): 7\nOpponent goals conceded (season avg per GW): 0.8\nPrice: 4.3"
    },
    {
        "role": "assistant",
        "content": '{"predicted_pts": 1.5, "confidence": 70, "reasoning": "Tough away fixture (FDR 5) with virtually no attacking threat and a team conceding heavily makes a blank or subbed-off likely."}'
    },
]


def get_next_gameweek():
    with open(os.path.join(RAW_DIR, "gameweeks.json")) as f:
        gws = json.load(f)
    for gw in gws:
        if not gw["finished"]:
            return gw["id"]
    return gws[-1]["id"] + 1


def build_player_prompt(row):
    """Convert a player's feature row into a prompt string. Uses only columns available in next_gw_predictions.csv."""
    home_away = row.get("home_away", "H" if row.get("was_home", 1) == 1 else "A")
    home_away_full = "Home" if home_away == "H" else "Away"
    opponent = row.get("opponent", "Unknown")
    team = row.get("team_name", "Unknown")

    lines = [
        f"Player: {row['player_name']} ({row['position']})",
        f"Team: {team}",
        f"Gameweek: {int(row['gameweek'])}",
        f"Fixture: {home_away_full} vs {opponent} (FDR: {int(row['fixture_difficulty'])})",
        f"Form (last 3 GW avg): {row['form_3gw']:.1f} pts",
        f"Minutes (last 3 GW avg): {row['avg_minutes_3gw']:.1f}",
        f"ICT index (last 3 GW avg): {row['ict_index_3gw']:.1f}",
        f"Price: {row['price']:.1f}",
    ]

    # Add optional columns if they exist
    for col, label, fmt in [
        ("form_5gw", "Form (last 5 GW avg)", ".1f"),
        ("goals_per_90_season", "Goals per 90 (season)", ".2f"),
        ("assists_per_90_season", "Assists per 90 (season)", ".2f"),
        ("clean_sheets_pct_season", "Clean sheet % (season)", ".1%"),
        ("bonus_avg_3gw", "Bonus avg (last 3 GW)", ".1f"),
        ("team_goals_scored_3gw", "Team goals scored (last 3 GW)", ".0f"),
        ("team_goals_conceded_3gw", "Team goals conceded (last 3 GW)", ".0f"),
        ("opponent_goals_conceded_season", "Opponent goals conceded (season avg per GW)", ".1f"),
    ]:
        if col in row.index and pd.notna(row[col]):
            lines.append(f"{label}: {format(row[col], fmt)}")

    # Add other models' predictions as context
    lines.append(f"\nOther model predictions for reference:")
    lines.append(f"  XGBoost: {row['xgb_predicted_pts']:.1f} pts (confidence: {row['xgb_confidence']}%)")
    if "auto_xgb_predicted_pts" in row.index and pd.notna(row.get("auto_xgb_predicted_pts")):
        lines.append(f"  Auto-Research XGBoost: {row['auto_xgb_predicted_pts']:.1f} pts (confidence: {row['auto_xgb_confidence']}%)")
    lines.append(f"  Fine-tuned LLM: {row['llm_predicted_pts']:.1f} pts (confidence: {row['llm_confidence']}%)")

    return "\n".join(lines)


def parse_claude_response(text):
    """Parse JSON from Claude's response."""
    # Try direct JSON parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON in the response
    match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def predict_batch(client, players_df, batch_size=20):
    """
    Send players to Claude in batches for efficiency.
    Each batch sends multiple players in one API call.
    """
    results = []
    total = len(players_df)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = players_df.iloc[batch_start:batch_end]

        # Build a multi-player prompt
        player_prompts = []
        for i, (_, row) in enumerate(batch.iterrows()):
            player_prompts.append(f"--- Player {i+1} ---\n{build_player_prompt(row)}")

        batch_prompt = (
            f"Predict FPL points for each of these {len(batch)} players. "
            f"Respond with one JSON object per player, each on its own line.\n\n"
            + "\n\n".join(player_prompts)
        )

        messages = list(FEW_SHOT_EXAMPLES) + [{"role": "user", "content": batch_prompt}]

        try:
            response = client.messages.create(
                model=PREDICTION_MODEL,
                max_tokens=2000,
                system=PREDICTION_SYSTEM,
                messages=messages,
            )
            response_text = response.content[0].text

            # Parse each JSON line
            parsed_results = []
            for line in response_text.strip().split("\n"):
                line = line.strip()
                if not line or line.startswith("---"):
                    continue
                parsed = parse_claude_response(line)
                if parsed:
                    parsed_results.append(parsed)

            # Match parsed results to players
            for i, (_, row) in enumerate(batch.iterrows()):
                if i < len(parsed_results):
                    r = parsed_results[i]
                    results.append({
                        "player_id": row["player_id"],
                        "predicted_pts": float(r.get("predicted_pts", 2.0)),
                        "confidence": int(r.get("confidence", 50)),
                        "reasoning": r.get("reasoning", ""),
                    })
                else:
                    results.append({
                        "player_id": row["player_id"],
                        "predicted_pts": 2.0,
                        "confidence": 30,
                        "reasoning": "Failed to parse prediction",
                    })

        except Exception as e:
            print(f"  ! Batch error: {e}")
            for _, row in batch.iterrows():
                results.append({
                    "player_id": row["player_id"],
                    "predicted_pts": 2.0,
                    "confidence": 30,
                    "reasoning": f"API error: {str(e)[:50]}",
                })

        print(f"  {batch_end}/{total} predictions done")
        time.sleep(0.5)  # rate limiting

    return results


def predict_individually(client, players_df):
    """Send each player individually for more reliable parsing."""
    results = []
    total = len(players_df)

    for i, (_, row) in enumerate(players_df.iterrows()):
        prompt = build_player_prompt(row)
        messages = list(FEW_SHOT_EXAMPLES) + [{"role": "user", "content": prompt}]

        try:
            response = client.messages.create(
                model=PREDICTION_MODEL,
                max_tokens=200,
                system=PREDICTION_SYSTEM,
                messages=messages,
            )
            parsed = parse_claude_response(response.content[0].text)

            if parsed:
                results.append({
                    "player_id": row["player_id"],
                    "predicted_pts": float(parsed.get("predicted_pts", 2.0)),
                    "confidence": int(parsed.get("confidence", 50)),
                    "reasoning": parsed.get("reasoning", ""),
                })
            else:
                results.append({
                    "player_id": row["player_id"],
                    "predicted_pts": 2.0,
                    "confidence": 30,
                    "reasoning": "Failed to parse response",
                })

        except Exception as e:
            print(f"  ! Error for {row['player_name']}: {e}")
            results.append({
                "player_id": row["player_id"],
                "predicted_pts": 2.0,
                "confidence": 30,
                "reasoning": f"API error",
            })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{total} predictions done")

        time.sleep(0.2)

    return results


def analyze_transfers(client, predictions_df, current_squad=None):
    """Use Claude Sonnet to analyze transfer recommendations."""
    # Get top and bottom players
    top_20 = predictions_df.nlargest(20, "claude_predicted_pts")
    bottom_20 = predictions_df.nsmallest(20, "claude_predicted_pts")

    top_summary = "\n".join([
        f"  {r['player_name']} ({r['position']}, {r['team_name']}, "
        f"£{r['price']:.1f}m) — XGB: {r['xgb_predicted_pts']:.1f}, "
        f"LLM: {r['llm_predicted_pts']:.1f}, Claude: {r['claude_predicted_pts']:.1f} "
        f"(conf: {r['claude_confidence']}%) — {r['claude_reasoning']}"
        for _, r in top_20.iterrows()
    ])

    prompt = f"""Based on FPL predictions for the upcoming gameweek, analyze the best transfer options.

TOP 20 PREDICTED PLAYERS:
{top_summary}

Please provide:
1. **Best Captain Pick** — who to captain and why (consider ceiling, not just floor)
2. **Best Budget Pick** — best player under £6.0m
3. **Best Premium Pick** — best player over £10.0m
4. **Best Differential** — a player owned by <10% who could haul
5. **Players to Avoid** — who looks overpredicted or has hidden risks

Respond in clear bullet points. Be specific and justify each pick."""

    response = client.messages.create(
        model=TRANSFER_MODEL,
        max_tokens=1000,
        system="You are an expert FPL manager with deep knowledge of Premier League football. Give specific, actionable advice.",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfers", action="store_true", help="Also run transfer analysis")
    parser.add_argument("--batch", action="store_true", help="Use batch mode (faster, less reliable)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set. Export it or pass as env var.")
        return
    client = Anthropic(api_key=api_key)
    next_gw = get_next_gameweek()
    print(f"Claude predictions for Gameweek {next_gw}")
    print(f"Using model: {PREDICTION_MODEL}\n")

    # Load the existing predictions (has player features + XGB/LLM predictions)
    existing = pd.read_csv(PREDICTIONS_PATH)
    print(f"Loaded {len(existing)} players from existing predictions\n")

    # Build feature data for Claude
    # We need the full feature rows — load from features.csv and build next GW
    # But we can reuse the existing predictions file which has the features we need
    features_for_claude = existing.copy()

    print("Running Claude predictions...")
    if args.batch:
        claude_results = predict_batch(client, features_for_claude, batch_size=15)
    else:
        claude_results = predict_individually(client, features_for_claude)

    # Save Claude-specific results first
    claude_df = pd.DataFrame(claude_results)
    claude_df.to_csv(OUTPUT_PATH, index=False)

    # Drop any existing claude columns from the predictions file before merging
    claude_cols = [c for c in existing.columns if "claude" in c.lower()]
    existing_clean = existing.drop(columns=claude_cols, errors="ignore")

    claude_df = claude_df.rename(columns={
        "predicted_pts": "claude_predicted_pts",
        "confidence": "claude_confidence",
        "reasoning": "claude_reasoning",
    })
    merged = existing_clean.merge(claude_df, on="player_id", how="left")

    # Compute Claude value score
    merged["claude_value_score"] = np.round(
        merged["claude_predicted_pts"] * (merged["claude_confidence"] / 100), 2
    ).astype(float)

    # Update combined value score to include all available models
    value_cols = ["xgb_value_score", "llm_value_score", "claude_value_score"]
    if "auto_xgb_value_score" in merged.columns:
        value_cols.append("auto_xgb_value_score")
    merged["combined_value_score"] = np.round(
        merged[value_cols].sum(axis=1) / len(value_cols),
        2,
    )

    # Re-sort by combined value
    merged = merged.sort_values("combined_value_score", ascending=False).reset_index(drop=True)

    # Save
    merged.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nPredictions saved to {PREDICTIONS_PATH}")

    # Also save Claude-specific results
    claude_df.to_csv(OUTPUT_PATH, index=False)

    # Summary
    print(f"\n{'='*80}")
    print(f"CLAUDE PREDICTION SUMMARY — GW {next_gw}")
    print(f"{'='*80}")
    print(f"Players predicted: {len(claude_results)}")
    valid = [r for r in claude_results if r['confidence'] > 30]
    pts = [r['predicted_pts'] for r in valid]
    print(f"Prediction range: {min(pts):.1f} to {max(pts):.1f}")
    print(f"Mean prediction: {np.mean(pts):.1f}")
    print(f"Avg confidence: {np.mean([r['confidence'] for r in valid]):.0f}%")

    has_auto = "auto_xgb_predicted_pts" in merged.columns
    auto_hdr = " {'Auto':>6}" if has_auto else ""
    auto_col = f"{'Auto':>6} " if has_auto else ""
    print(f"\n{'Player':<16} {'XGB':>6} {auto_col}{'LLM':>6} {'Claude':>6} {'Conf':>5} {'Reasoning'}")
    print(f"{'-'*90}")
    for _, r in merged.head(15).iterrows():
        reason = r.get('claude_reasoning', '')[:35] if pd.notna(r.get('claude_reasoning')) else ''
        auto_str = f"{r['auto_xgb_predicted_pts']:>6.1f} " if has_auto else ""
        print(
            f"{r['player_name']:<16} "
            f"{r['xgb_predicted_pts']:>6.1f} "
            f"{auto_str}"
            f"{r['llm_predicted_pts']:>6.1f} "
            f"{r['claude_predicted_pts']:>6.1f} "
            f"{r['claude_confidence']:>4.0f}% "
            f"{reason}"
        )

    # Transfer analysis
    if args.transfers:
        print(f"\n{'='*80}")
        print("TRANSFER ANALYSIS (Claude Sonnet)")
        print(f"{'='*80}\n")
        analysis = analyze_transfers(client, merged)
        print(analysis)

        # Save analysis
        with open(os.path.join(ROOT, "eval", "claude_transfer_analysis.txt"), "w") as f:
            f.write(analysis)
        print(f"\nAnalysis saved to eval/claude_transfer_analysis.txt")


if __name__ == "__main__":
    main()
