# Gameweek Results Tracker

How do our four prediction models perform when it actually matters — picking an FPL starting XI that scores real points?

This document tracks the live results of each model's "Best Predicted XI" against actual FPL outcomes, gameweek by gameweek. Unlike our [evaluation framework](eval/) which tests on historical data, this is the real thing — predictions made *before* the deadline, scored *after* the matches.

---

## How We Score

For each gameweek, every model independently selects its optimal XI:

- **Budget:** 100m (same as real FPL)
- **Constraints:** Max 3 players per team, valid formation (GK + 3-5 DEF + 2-5 MID + 1-3 FWD)
- **Captain:** Highest value-score player gets 2x points
- **Vice-captain:** Gets 2x only if the captain plays 0 minutes (full FPL rules)
- **Selection method:** Greedy algorithm using each model's own value scores (predicted points x confidence)

Each model picks independently — they often select different players, formations, and captains.

### Metrics

| Metric | What it measures |
|:---|:---|
| **Total Points** | Actual FPL points scored by the model's XI (with captain bonus) — higher is better |
| **MAE** | Mean Absolute Error per player (predicted vs actual) — lower is better |
| **Delta** | Total predicted minus total actual — closer to 0 is better (positive = overconfident) |

---

## Cumulative Standings

| Rank | Model | Total Points | Avg MAE | GWs Played |
|:---|:---|:---|:---|:---|
| 1 | **Claude Haiku** | **53** | **4.20** | 1 |
| 2 | LLM (Llama 3.2 3B) | 44 | 4.81 | 1 |
| 3 | Auto-Research XGBoost | 42 | 3.78 | 1 |
| 4 | XGBoost (Original) | 18 | 1.95 | 1 |

> After just one gameweek, these standings are preliminary. The real picture will emerge over 5-10 gameweeks as variance smooths out.

---

## Gameweek 32

**Date:** 11-14 April 2026 | **First evaluated gameweek**

### Scoreboard

| Model | Formation | Actual Points | Predicted Points | Delta | MAE | Captain | Captain Pts |
|:---|:---|---:|---:|---:|---:|:---|---:|
| **Claude Haiku** | 3-5-2 | **53** | 65.2 | +12.2 | 4.20 | B.Fernandes (C) | 4 x2 = 8 |
| LLM | 3-5-2 | 44 | 77.7 | +33.7 | 4.81 | B.Fernandes (C) | 4 x2 = 8 |
| Auto-Research | 4-4-2 | 42 | 57.0 | +15.0 | 3.78 | B.Fernandes (C) | 4 x2 = 8 |
| XGBoost | 4-4-2 | 18 | 37.4 | +19.4 | 1.95 | Bradley (C) | 0 x2 = 0 |

### Key Observations

**Claude wins on points (53), Auto-Research wins on accuracy (MAE 3.78)**

The two metrics tell different stories. Claude Haiku assembled the highest-scoring XI by picking premium players like Salah (8 pts), Virgil van Dijk (9 pts), and Bowen (14 pts). Auto-Research had the lowest per-player prediction error, but its squad of budget picks scored fewer total points.

**XGBoost's captain disaster**

XGBoost picked Conor Bradley (LIV, DEF) as captain — a player who didn't even play (0 minutes, 0 points). The vice-captain rule kicked in, giving Palhinha 2x his 1 point. This single decision cost XGBoost massively. Meanwhile, every other model captained Bruno Fernandes, who scored 4 points (8 with the captain bonus). Captain selection is critical.

**XGBoost's low MAE is misleading**

XGBoost has the lowest MAE (1.95) because it predicted conservatively — every player between 3.0-4.0 predicted points. When actual scores are mostly 0-4, predicting ~3 for everyone produces low error but also low total points. It's the "predict 2 for everyone" problem from a different angle.

**Bowen was the week's hero — only 3 models caught him**

Jarrod Bowen (WHU) scored 14 points — the highest in any model's XI. Claude, LLM, and Auto-Research all selected him. XGBoost missed him entirely. However, no model predicted anywhere near 14 points for him (predictions ranged from 3.4 to 6.4).

**All models overestimated**

Every model predicted higher totals than reality. The LLM was the most overconfident (Delta +33.7), while Claude was closest (Delta +12.2). This is typical in FPL — models tend to predict optimistic baselines, and many players disappoint.

### Per-Model Best XI

#### Claude Haiku — 53 pts (Winner)

| Player | Pos | Team | Predicted | Actual | Effective |
|:---|:---|:---|---:|---:|---:|
| Bowen | FWD | WHU | 5.8 | 14 | 14 |
| Virgil | DEF | LIV | 5.1 | 9 | 9 |
| B.Fernandes (C) | MID | MUN | 7.8 | 4 | 8 |
| M.Salah | MID | LIV | 5.1 | 8 | 8 |
| Szoboszlai | MID | LIV | 5.8 | 5 | 5 |
| Gabriel (V) | DEF | ARS | 7.3 | 4 | 4 |
| Henderson | GK | CRY | 5.2 | 2 | 2 |
| Joao Pedro | FWD | CHE | 6.1 | 2 | 2 |
| Cunha | MID | MUN | 5.4 | 1 | 1 |
| Dorgu | DEF | MUN | 5.8 | 0 | 0 |
| Saka | MID | ARS | 5.8 | 0 | 0 |

#### LLM (Llama 3.2 3B) — 44 pts

| Player | Pos | Team | Predicted | Actual | Effective |
|:---|:---|:---|---:|---:|---:|
| Bowen | FWD | WHU | 6.4 | 14 | 14 |
| B.Fernandes (C) | MID | MUN | 9.5 | 4 | 8 |
| Van Hecke | DEF | BHA | 6.1 | 5 | 5 |
| Szoboszlai | MID | LIV | 6.6 | 5 | 5 |
| Gabriel | DEF | ARS | 7.4 | 4 | 4 |
| Henderson | GK | CRY | 6.6 | 2 | 2 |
| Semenyo | MID | MCI | 6.7 | 2 | 2 |
| Joao Pedro (V) | FWD | CHE | 7.7 | 2 | 2 |
| Mac Allister | MID | LIV | 6.6 | 1 | 1 |
| Sarr | MID | CRY | 6.5 | 1 | 1 |
| Dorgu | DEF | MUN | 7.6 | 0 | 0 |

#### Auto-Research XGBoost — 42 pts

| Player | Pos | Team | Predicted | Actual | Effective |
|:---|:---|:---|---:|---:|---:|
| Bowen | FWD | WHU | 3.4 | 14 | 14 |
| B.Fernandes (C) | MID | MUN | 12.5 | 4 | 8 |
| Darlow | GK | LEE | 3.6 | 4 | 4 |
| Gabriel (V) | DEF | ARS | 7.1 | 4 | 4 |
| Sangare | MID | NFO | 3.7 | 4 | 4 |
| Ampadu | MID | LEE | 4.5 | 3 | 3 |
| Gibbs-White | MID | NFO | 4.7 | 2 | 2 |
| Igor Jesus | FWD | NFO | 3.8 | 2 | 2 |
| Cash | DEF | AVL | 4.7 | 1 | 1 |
| Dorgu | DEF | MUN | 5.2 | 0 | 0 |
| Rodon | DEF | LEE | 3.8 | 0 | 0 |

#### XGBoost (Original) — 18 pts

| Player | Pos | Team | Predicted | Actual | Effective |
|:---|:---|:---|---:|---:|---:|
| Andre | MID | WOL | 3.5 | 4 | 4 |
| Collins | DEF | BRE | 3.0 | 3 | 3 |
| Garner | MID | EVE | 3.6 | 3 | 3 |
| J.Palhinha (V) | MID | TOT | 3.6 | 1 | 2 |
| Sesko | FWD | MUN | 4.0 | 2 | 2 |
| Strand Larsen | FWD | CRY | 3.4 | 2 | 2 |
| Jose Sa | GK | WOL | 3.6 | 1 | 1 |
| Mitoma | MID | BHA | 3.2 | 1 | 1 |
| Bradley (C) | DEF | LIV | 3.1 | 0 | 0 |
| Ajer | DEF | BRE | 3.3 | 0 | 0 |
| Henry | DEF | BRE | 3.1 | 0 | 0 |

---

## What We Expect Going Forward

After one gameweek, we expect the following patterns to emerge (or be disproven) over the coming weeks:

1. **Points vs MAE will diverge.** Models that pick premium/high-ceiling players (Claude, LLM) should score more total points but have higher MAE. Conservative models (XGBoost) will have lower MAE but fewer points. The question is which approach wins the FPL game — you need points, not low error.

2. **Captain selection will be the biggest swing factor.** The difference between XGBoost (0 captain points) and everyone else (8 captain points) was 8 points from a single decision. Over a season, captain picks alone could determine the best model.

3. **Auto-Research should stabilise as the most balanced.** It had the best MAE (3.78) while still scoring competitively (42 pts). Its two-stage architecture (classifier + regressor) should help it avoid selecting players who don't play.

4. **The LLM's overconfidence needs watching.** A delta of +33.7 means the LLM predicted nearly double the actual points. If this persists, it might pick flashy players who consistently disappoint.

5. **One gameweek is noise.** A single GW32 could be an outlier. We need 5-10 gameweeks before drawing real conclusions. The cumulative table above will tell the true story.

---

## How to View Results in the App

The Streamlit dashboard includes a **Results** tab with interactive pitch views for each model:

```bash
streamlit run ui/app.py
# Navigate to the "Results" tab
```

Each model's XI is displayed on a football pitch with actual points, captain/vice badges, and predicted-vs-actual deltas per player.

---

*This document is updated after each evaluated gameweek. Results are generated by the evaluation logic in [`ui/results_logic.py`](ui/results_logic.py) using archived predictions from [`data/predictions/`](data/predictions/) and actual FPL data from [`data/raw/player_histories/`](data/raw/player_histories/).*
