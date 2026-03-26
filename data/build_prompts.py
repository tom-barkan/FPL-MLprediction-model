"""
Convert the feature matrix into prompt-completion pairs for LLM training.

Produces two formats:
  1. Simple JSONL (prompt/completion) → data/processed/train_prompts.jsonl, test_prompts.jsonl
  2. MLX chat format (messages) → data/mlx/train.jsonl, valid.jsonl, test.jsonl

Train/test split: GW 4-30 = train, GW 31+ = test
Validation (for MLX): GW 28-30 sliced from training data
"""

import json
import os
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
MLX_DIR = os.path.join(os.path.dirname(__file__), "mlx")

SYSTEM_PROMPT = (
    "You are an FPL points predictor. Given player stats and fixture info, "
    "predict the total FPL points for the gameweek. Respond with only a single integer."
)


def row_to_user_prompt(row):
    """Convert a feature row into a natural language prompt."""
    home_away = "Home" if row["was_home"] == 1 else "Away"
    lines = [
        f"Player: {row['player_name']} ({row['position']})",
        f"Gameweek: {int(row['gameweek'])}",
        f"Fixture: {home_away} (FDR: {int(row['fixture_difficulty'])})",
        f"Form (last 3 GW avg): {row['form_3gw']:.1f} pts",
        f"Form (last 5 GW avg): {row['form_5gw']:.1f} pts",
        f"Minutes (last 3 GW avg): {row['avg_minutes_3gw']:.1f}",
        f"Goals per 90 (season): {row['goals_per_90_season']:.2f}",
        f"Assists per 90 (season): {row['assists_per_90_season']:.2f}",
        f"ICT index (last 3 GW avg): {row['ict_index_3gw']:.1f}",
        f"Clean sheet % (season): {row['clean_sheets_pct_season']:.1%}",
        f"Bonus avg (last 3 GW): {row['bonus_avg_3gw']:.1f}",
        f"Team goals scored (last 3 GW): {row['team_goals_scored_3gw']:.0f}",
        f"Team goals conceded (last 3 GW): {row['team_goals_conceded_3gw']:.0f}",
        f"Opponent goals conceded (season avg per GW): {row['opponent_goals_conceded_season']:.1f}",
        f"Price: {row['price']:.1f}",
    ]
    return "\n".join(lines)


def build_simple_prompts(df, output_path):
    """Build simple prompt/completion JSONL."""
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            user_prompt = row_to_user_prompt(row)
            prompt = (
                "Predict the FPL points for the following player this gameweek.\n\n"
                f"{user_prompt}\n\n"
                "Predict the points total as a single integer."
            )
            record = {
                "prompt": prompt,
                "completion": str(int(row["target_points"])),
            }
            f.write(json.dumps(record) + "\n")

    print(f"  → {len(df)} prompts written to {output_path}")


def build_mlx_prompts(df, output_path):
    """Build MLX chat-format JSONL (system/user/assistant messages)."""
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row_to_user_prompt(row)},
                    {"role": "assistant", "content": str(int(row["target_points"]))},
                ]
            }
            f.write(json.dumps(record) + "\n")

    print(f"  → {len(df)} prompts written to {output_path}")


def balance_training_data(df):
    """
    Balance training data at the individual point level so the model
    learns to predict the full range of scores (0-10+), not just the mode.

    Each point value gets roughly equal representation.
    """
    df = df.copy()

    # Clip to buckets: each individual point value 0 through 10, plus 11+
    df["_pts_bucket"] = df["target_points"].clip(lower=0, upper=11)

    bucket_counts = df["_pts_bucket"].value_counts().sort_index()
    target_per_bucket = int(bucket_counts.median())

    print(f"\nBalancing training data (per-point-value):")
    print(f"  Before: {dict(bucket_counts)}")
    print(f"  Target per bucket: ~{target_per_bucket}")

    balanced_parts = []
    for pts_val, group in df.groupby("_pts_bucket"):
        if len(group) >= target_per_bucket:
            # Downsample but keep at least target_per_bucket
            balanced_parts.append(group.sample(n=target_per_bucket, random_state=42))
        else:
            # Oversample to reach target
            repeats = target_per_bucket // len(group)
            remainder = target_per_bucket % len(group)
            parts = [group] * repeats
            if remainder > 0:
                parts.append(group.sample(n=remainder, random_state=42))
            balanced_parts.append(pd.concat(parts))

    result = pd.concat(balanced_parts).drop(columns=["_pts_bucket"])
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  After: {len(result)} rows")
    after_dist = result["target_points"].clip(lower=0, upper=11).value_counts().sort_index()
    print(f"  Distribution: {dict(after_dist)}")

    return result


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MLX_DIR, exist_ok=True)

    features_path = os.path.join(PROCESSED_DIR, "features.csv")
    if not os.path.exists(features_path):
        print("features.csv not found. Run build_features.py first.")
        return

    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} rows from features.csv")

    # Split: train = GW 4-30, test = GW 31+
    train_full = df[df["gameweek"] <= 30]
    test = df[df["gameweek"] > 30]

    # Validation split for MLX: GW 28-30 from training data
    train_mlx_raw = train_full[train_full["gameweek"] <= 27]
    valid_mlx = train_full[train_full["gameweek"].between(28, 30)]

    # Balance the MLX training set: oversample underrepresented point values
    # so the model learns to differentiate rather than always predicting the mode
    train_mlx = balance_training_data(train_mlx_raw)

    print(f"\nSplit sizes:")
    print(f"  Train (GW 4-27 for MLX, balanced): {len(train_mlx)} rows")
    print(f"  Valid (GW 28-30 for MLX): {len(valid_mlx)} rows")
    print(f"  Train full (GW 4-30):    {len(train_full)} rows")
    print(f"  Test (GW 31+):           {len(test)} rows")

    # Simple JSONL format
    print("\nBuilding simple prompts...")
    build_simple_prompts(train_full, os.path.join(PROCESSED_DIR, "train_prompts.jsonl"))
    build_simple_prompts(test, os.path.join(PROCESSED_DIR, "test_prompts.jsonl"))

    # MLX chat format (balanced training data)
    print("\nBuilding MLX chat prompts (balanced)...")
    build_mlx_prompts(train_mlx, os.path.join(MLX_DIR, "train.jsonl"))
    build_mlx_prompts(valid_mlx, os.path.join(MLX_DIR, "valid.jsonl"))
    build_mlx_prompts(test, os.path.join(MLX_DIR, "test.jsonl"))

    print("\nDone! All prompt files generated.")


if __name__ == "__main__":
    main()
