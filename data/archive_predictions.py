"""
Archive current GW predictions before they get overwritten.

Copies data/processed/next_gw_predictions.csv to data/predictions/gw{N}_predictions.csv.
Run: python data/archive_predictions.py
"""

import os
import shutil

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_PATH = os.path.join(ROOT, "data", "processed", "next_gw_predictions.csv")
ARCHIVE_DIR = os.path.join(ROOT, "data", "predictions")


def archive():
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"No predictions file found at {PREDICTIONS_PATH}")
        return

    df = pd.read_csv(PREDICTIONS_PATH)
    gw = int(df["gameweek"].iloc[0])

    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    dest = os.path.join(ARCHIVE_DIR, f"gw{gw}_predictions.csv")
    if os.path.exists(dest):
        print(f"Archive already exists: {dest} — skipping")
        return

    shutil.copy2(PREDICTIONS_PATH, dest)
    print(f"Archived GW{gw} predictions to {dest}")


if __name__ == "__main__":
    archive()
