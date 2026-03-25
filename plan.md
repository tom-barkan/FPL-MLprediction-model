# FPL Points Prediction: Fine-Tuned LLM vs XGBoost

## Project Goal

Build two competing models that predict Fantasy Premier League player points per gameweek, then evaluate which approach wins and why. The LLM runs locally on Mac Mini M4 using open-source tools. The outcome is a written assessment comparing both approaches — useful as product leadership portfolio material.

---

## Architecture Overview

```
FPL API → Data Pipeline → Training Data
                              ├── Tabular CSV → XGBoost Model ──────┐
                              └── JSONL Prompts → Fine-Tuned LLM ───┤
                                                                     ▼
                                                              Evaluation Harness
                                                                     │
                                                              Streamlit Comparison UI
                                                                     │
                                                              GitHub Repository
                                                              ├── Notebooks (documented experiments)
                                                              ├── README (results + takeaways)
                                                              └── Tagged releases per experiment
```

---

## Phase 1: Data Pipeline (Week 1–2)

### 1.1 Understand the FPL API

The FPL API is free, unauthenticated, and returns JSON.

**Key endpoints:**

| Endpoint | What it gives you |
|----------|-------------------|
| `/api/bootstrap-static/` | All players, teams, gameweeks, positions — the master dataset |
| `/api/element-summary/{player_id}/` | Per-player history: every gameweek they've played, fixtures, past seasons |
| `/api/fixtures/` | All fixtures with scores, difficulty ratings |

**Rate limiting:** Be polite — add 1-second delays between requests. No auth token needed.

### 1.2 Build the Data Fetcher

Create `data/fetch_fpl.py`:

```bash
mkdir -p fpl-predictor/{data,models,eval,ui}
cd fpl-predictor
python -m venv .venv
source .venv/bin/activate
pip install requests pandas scikit-learn xgboost streamlit matplotlib
```

**What to fetch and store:**

1. **Player master list** from bootstrap-static: `id, web_name, team, position, now_cost`
2. **Gameweek history** for every player via element-summary: `round, total_points, minutes, goals_scored, assists, clean_sheets, bonus, bps, influence, creativity, threat, ict_index, was_home, opponent_team`
3. **Fixture data**: `team_h, team_a, team_h_score, team_a_score, team_h_difficulty, team_a_difficulty`

**Store as:**
- `data/raw/players.json` — master player list
- `data/raw/player_histories/` — one JSON per player
- `data/raw/fixtures.json` — all fixtures

### 1.3 Feature Engineering

Create `data/build_features.py` that produces a single CSV with one row per player-gameweek:

**Core features (per row):**

| Feature | Description |
|---------|-------------|
| `player_id` | FPL player ID |
| `player_name` | Web name |
| `position` | GK / DEF / MID / FWD |
| `gameweek` | GW number (1–38) |
| `was_home` | 1 if home, 0 if away |
| `opponent_team` | Opponent team ID |
| `fixture_difficulty` | FDR rating (1–5) |
| `form_3gw` | Average points over last 3 GWs |
| `form_5gw` | Average points over last 5 GWs |
| `avg_minutes_3gw` | Average minutes last 3 GWs (proxy for nailedness) |
| `goals_per_90_season` | Goals per 90 minutes this season so far |
| `assists_per_90_season` | Assists per 90 this season |
| `ict_index_3gw` | Average ICT index last 3 GWs |
| `clean_sheets_pct_season` | % of GWs with clean sheet (DEF/GK) |
| `bonus_avg_3gw` | Average bonus points last 3 GWs |
| `team_goals_scored_3gw` | Team's goals scored last 3 GWs |
| `team_goals_conceded_3gw` | Team's goals conceded last 3 GWs |
| `opponent_goals_conceded_season` | How leaky is the opposition? |
| `price` | Current price (÷10 for display) |
| `target_points` | **THE LABEL** — actual points scored that GW |

**Important rules:**
- Never leak future data — all rolling features must use only prior gameweeks
- Skip GW1–3 for training since rolling averages are unreliable
- Drop rows where `minutes == 0` in the target GW (player didn't play — not a prediction problem)

**Output:** `data/processed/features.csv`

### 1.4 Build LLM Training Prompts

Create `data/build_prompts.py` that converts each row into a natural language prompt-completion pair in JSONL:

```json
{
  "prompt": "Predict the FPL points for the following player this gameweek.\n\nPlayer: Salah (MID)\nGameweek: 20\nFixture: Home vs Burnley (FDR: 2)\nForm (last 3 GW avg): 8.3 pts\nMinutes (last 3 GW avg): 90.0\nGoals per 90 (season): 0.62\nAssists per 90 (season): 0.31\nICT index (last 3 GW avg): 42.1\nBonus avg (last 3 GW): 1.7\nTeam goals scored (last 3 GW): 9\nTeam goals conceded (last 3 GW): 2\nOpponent goals conceded (season avg per GW): 1.8\nPrice: 13.2\n\nPredict the points total as a single integer.",
  "completion": "9"
}
```

**Output:** `data/processed/train_prompts.jsonl` and `data/processed/test_prompts.jsonl`

### 1.5 Train/Test Split Strategy

- **Training set:** GW 4–30 (all players, all seasons you have data for)
- **Test set:** GW 31–38 (held out — never seen during training)
- This simulates real usage: predicting the end of season based on what you've learned from the season so far
- Split both the CSV and the JSONL using the same GW boundary

### 1.6 Manual Data Labelling Exercise (Prediction Confidence)

This sub-task exists purely to teach you why labelling is the hardest part of real AI product work. The FPL API gives you clean, pre-structured data — that's not representative of what your AI teams will face.

**The task:** Manually label 100 player-gameweek rows from your test set with a "prediction confidence" score:

| Score | Meaning |
|-------|---------|
| 1 — High confidence | Strong form, easy fixture, nailed starter — you'd confidently predict a narrow range |
| 2 — Medium confidence | Mixed signals — decent form but tough fixture, or good fixture but inconsistent |
| 3 — Low confidence | Rotation risk, returning from injury, newly promoted team, wildcard factor |

**How to do it:**

1. Sample 100 rows from the test set (stratified across positions and fixture difficulties)
2. Open them in a spreadsheet
3. For each row, read the features and assign a confidence score based on your FPL knowledge
4. Time yourself — note how long 100 labels takes
5. After labelling, re-label 20 random rows a week later and check your own consistency

**Save as:** `data/labelled/confidence_labels.csv`

**What you'll learn:**

- **Labelling is slow.** 100 rows will take 30–60 minutes. Multiply that by the thousands you'd need for production. Now you understand why labelling budgets matter.
- **Labelling is subjective.** Your re-labelling consistency won't be perfect — you'll disagree with yourself on borderline cases. This is the inter-annotator agreement problem that plagues every AI team.
- **Labels reveal data gaps.** You'll notice cases where the features don't capture what you intuitively know (e.g. "this player is about to lose their starting spot to a new signing" — that's not in the API data). This teaches you about feature coverage.
- **Confidence labels are product-useful.** Later, you can use these to stratify your eval results: "the model is accurate on high-confidence cases but terrible on low-confidence ones." That's a product insight — maybe you only show predictions when confidence is high.

**Use in eval (Phase 4):** Add an eval category that breaks down model accuracy by your confidence labels. This tells you whether the models struggle on the same cases you'd struggle on as a human.

---

## Phase 2: XGBoost Baseline (Week 3)

### 2.1 Train the Model

Create `models/train_xgboost.py`:

```python
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

# Load
df = pd.read_csv("data/processed/features.csv")
train = df[df["gameweek"] <= 30]
test = df[df["gameweek"] > 30]

# Features (drop non-feature columns)
drop_cols = ["player_id", "player_name", "target_points", "gameweek"]
X_train = train.drop(columns=drop_cols)
y_train = train["target_points"]
X_test = test.drop(columns=drop_cols)
y_test = test["target_points"]

# Encode categoricals
X_train = pd.get_dummies(X_train, columns=["position"])
X_test = pd.get_dummies(X_test, columns=["position"])

# Train
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
    eval_metric="mae"
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

# Save
model.save_model("models/xgboost_fpl.json")

# Evaluate
preds = model.predict(X_test)
results = {
    "mae": mean_absolute_error(y_test, preds),
    "rmse": mean_squared_error(y_test, preds, squared=False),
}
print(results)
json.dump(results, open("eval/xgboost_results.json", "w"))
```

### 2.2 Analyse Feature Importance

```python
import matplotlib.pyplot as plt

xgb.plot_importance(model, max_num_features=15)
plt.tight_layout()
plt.savefig("eval/xgboost_feature_importance.png")
```

This tells you what the model actually relies on — form, ICT, fixture difficulty, etc. Compare this to your FPL intuition.

### 2.3 Expected Baseline Performance

For FPL points prediction, a well-tuned XGBoost typically achieves:
- **MAE: 1.5–2.5 points** (meaning on average it's off by ~2 points per player per GW)
- This is actually decent — FPL points are inherently noisy (a penalty, red card, or hat trick can swing 15+ points)

---

## Phase 3: Local LLM Fine-Tune (Week 4–5)

### 3.1 Set Up MLX on Mac Mini M4

```bash
pip install mlx mlx-lm
```

MLX is Apple's framework optimised for M-series silicon. It runs fine-tuning and inference on your unified memory without needing a discrete GPU.

### 3.2 Download Base Model

Use **Llama 3.2 3B** as your starting point — good balance of capability vs memory on M4:

```bash
# Using Hugging Face CLI
pip install huggingface_hub
huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit --local-dir models/llama-3.2-3b
```

The 4-bit quantised version uses ~2GB RAM. You can also try `Mistral-7B-Instruct-v0.3` (4-bit, ~4GB) if you want more capacity.

### 3.3 Prepare Data for MLX Fine-Tuning

MLX expects a specific JSONL chat format. Create `data/build_mlx_prompts.py`:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an FPL points predictor. Given player stats and fixture info, predict the total FPL points for the gameweek. Respond with only a single integer."
    },
    {
      "role": "user",
      "content": "Player: Salah (MID)\nGameweek: 20\nFixture: Home vs Burnley (FDR: 2)\nForm (last 3 GW avg): 8.3 pts\nMinutes (last 3 GW avg): 90.0\nGoals per 90 (season): 0.62\nAssists per 90 (season): 0.31\nICT index (last 3 GW avg): 42.1\nBonus avg (last 3 GW): 1.7\nTeam goals scored (last 3 GW): 9\nTeam goals conceded (last 3 GW): 2\nOpponent goals conceded (season avg per GW): 1.8\nPrice: 13.2"
    },
    {
      "role": "assistant",
      "content": "9"
    }
  ]
}
```

**Split files:**
- `data/mlx/train.jsonl`
- `data/mlx/valid.jsonl` (use a small slice of training data, e.g. GW 28–30)
- `data/mlx/test.jsonl`

### 3.4 Run LoRA Fine-Tuning

```bash
mlx_lm.lora \
  --model models/llama-3.2-3b \
  --data data/mlx \
  --train \
  --batch-size 4 \
  --lora-layers 8 \
  --iters 600 \
  --learning-rate 1e-5 \
  --adapter-path models/fpl-lora-adapter
```

**Key parameters to experiment with:**

| Parameter | Start | Range to try |
|-----------|-------|-------------|
| `--iters` | 600 | 300–1000 |
| `--learning-rate` | 1e-5 | 5e-6 to 5e-5 |
| `--lora-layers` | 8 | 4–16 |
| `--batch-size` | 4 | 2–8 (depends on RAM) |

Training ~600 iterations on 3B model should take **15–30 minutes** on M4.

### 3.5 Run Inference

Create `models/predict_llm.py`:

```bash
mlx_lm.generate \
  --model models/llama-3.2-3b \
  --adapter-path models/fpl-lora-adapter \
  --prompt "Player: Saka (MID)..."
```

For batch inference over the test set, write a Python script using the `mlx_lm` library:

```python
from mlx_lm import load, generate

model, tokenizer = load(
    "models/llama-3.2-3b",
    adapter_path="models/fpl-lora-adapter"
)

# Loop through test prompts, parse integer from response
# Save predictions to eval/llm_predictions.csv
```

**Critical: parse the output carefully.** The model might return "7" or "I predict 7 points" or garbage. Build a robust parser:
1. Try `int(response.strip())`
2. Fall back to regex for first integer in response
3. If no integer found, log as prediction failure

### 3.6 Prompt Engineering Baselines (Before Fine-Tuning)

Before you fine-tune anything, test how far prompting alone gets you. This is the most common real-world question: "do we actually need to fine-tune, or can we just prompt better?"

Run the base Llama 3.2 3B against the test set with three prompting strategies:

**Strategy A: Zero-Shot**
Just the system prompt and the player data. No examples. This is what you already have.

**Strategy B: Few-Shot (3–5 examples)**
Inject 3–5 real examples into the prompt before the actual question. Pick examples that cover different positions and outcomes:

```json
{
  "messages": [
    {"role": "system", "content": "You are an FPL points predictor..."},
    {"role": "user", "content": "Player: Palmer (MID)\nFixture: Home vs Wolves (FDR: 2)\nForm: 7.0..."},
    {"role": "assistant", "content": "8"},
    {"role": "user", "content": "Player: Gabriel (DEF)\nFixture: Away vs City (FDR: 5)\nForm: 3.2..."},
    {"role": "assistant", "content": "2"},
    {"role": "user", "content": "Player: Watkins (FWD)\nFixture: Home vs Everton (FDR: 3)\nForm: 4.5..."},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "[ACTUAL TEST CASE]"}
  ]
}
```

**Strategy C: Chain-of-Thought**
Ask the model to reason step by step before giving a number:

```
System: You are an FPL points predictor. Think step by step about the player's
form, fixture difficulty, and position before giving your prediction. End your
response with "Prediction: X" where X is a single integer.
```

**What to measure:**
- MAE for each strategy
- Parse failure rate (CoT will be messier to parse)
- Token count per prediction (cost proxy)

**Why this matters:**
If few-shot gets within 0.3 MAE of fine-tuning, that completely changes the build-vs-buy calculus. Few-shot requires zero training, no infrastructure, and works with any model — including API models. This is the kind of finding that changes how you'd advise a team on their AI approach.

Save results as `eval/prompt_baselines.csv` and include in the eval suite.

### 3.7 Also Test the Base Model (No Fine-Tuning)

Run the same test set through the base Llama 3.2 3B **without** the LoRA adapter. This gives you a five-way comparison: zero-shot vs few-shot vs chain-of-thought vs fine-tuned LLM vs XGBoost.

---

## Phase 4: Eval Framework (Week 5–6)

This is the most transferable phase of the project. Every AI product team struggles with evals. You're building one from scratch.

### 4.1 Eval Architecture

The eval system has three layers:

```
eval/
├── eval_suite.yaml              # Defines all eval cases and categories
├── run_eval.py                  # Single command: runs all models, scores everything
├── cases/                       # Named eval scenarios
│   ├── premium_home_easy.jsonl
│   ├── budget_away_hard.jsonl
│   ├── returning_from_injury.jsonl
│   ├── double_gameweek.jsonl
│   └── ...
├── checks/                      # Output quality validators
│   └── output_checks.py
├── results/                     # Timestamped results from every run
│   ├── 2026-03-25_v1-baseline/
│   │   ├── scores.json
│   │   ├── regressions.json
│   │   └── summary.md
│   └── 2026-03-28_v2-more-iters/
│       └── ...
├── baseline/                    # The "best so far" to compare against
│   └── scores.json
└── compare.py                   # Diff two runs, flag regressions
```

### 4.2 Define Eval Cases

This is the hard part and the most valuable. Each eval case is a named scenario that tests something specific about model behaviour.

Create `eval/eval_suite.yaml`:

```yaml
categories:
  fixture_context:
    description: "Does the model understand fixture difficulty?"
    cases:
      - name: premium_mid_home_easy
        description: "Top midfielder at home vs bottom-half team"
        criteria: "Should predict 6+ points"
        filter:
          position: MID
          price_min: 10.0
          was_home: true
          fixture_difficulty_max: 2
        expected_range: [5, 15]
        count: 30  # sample 30 matching rows from test set

      - name: premium_mid_away_hard
        description: "Top midfielder away vs top-6 team"
        criteria: "Should predict lower than home easy equivalent"
        filter:
          position: MID
          price_min: 10.0
          was_home: false
          fixture_difficulty_min: 4
        expected_range: [2, 8]
        count: 30

  positional:
    description: "Does the model handle different positions correctly?"
    cases:
      - name: goalkeeper_clean_sheet_likely
        description: "GK at home vs low-scoring opponent"
        filter:
          position: GK
          was_home: true
          opponent_goals_conceded_season_max: 1.2
        expected_range: [3, 8]
        count: 20

      - name: budget_defender_away
        description: "Cheap defender away from home"
        filter:
          position: DEF
          price_max: 4.5
          was_home: false
        expected_range: [1, 5]
        count: 20

      - name: premium_forward_form
        description: "Expensive striker in good form"
        filter:
          position: FWD
          price_min: 9.0
          form_3gw_min: 6.0
        expected_range: [4, 12]
        count: 20

  edge_cases:
    description: "How does the model handle unusual situations?"
    cases:
      - name: player_low_minutes
        description: "Player averaging <60 mins — rotation risk"
        criteria: "Should predict lower to account for sub/bench risk"
        filter:
          avg_minutes_3gw_max: 60
        expected_range: [0, 4]
        count: 20

      - name: player_on_hot_streak
        description: "Player with 3+ consecutive 7+ point returns"
        criteria: "Should predict high but not just extrapolate blindly"
        filter:
          form_3gw_min: 7.0
        expected_range: [4, 12]
        count: 20

      - name: newly_promoted_team
        description: "Player from newly promoted team"
        criteria: "Less historical data — how does model handle uncertainty?"
        filter:
          team_goals_scored_3gw_max: 3
          team_goals_conceded_3gw_min: 5
        expected_range: [1, 5]
        count: 15

  high_value:
    description: "The predictions that actually matter for FPL decisions"
    cases:
      - name: captain_pick_candidates
        description: "Players you'd consider captaining — the highest stakes prediction"
        filter:
          price_min: 10.0
          form_3gw_min: 5.0
          fixture_difficulty_max: 3
        expected_range: [5, 15]
        count: 30

      - name: differential_picks
        description: "Mid-price players with potential — where good predictions win mini-leagues"
        filter:
          price_min: 6.0
          price_max: 9.0
          ict_index_3gw_min: 30
        expected_range: [2, 10]
        count: 30

  human_confidence:
    description: "Accuracy stratified by your manual confidence labels from Phase 1.6"
    cases:
      - name: high_confidence_cases
        description: "Cases you labelled as easy to predict — model should perform best here"
        criteria: "MAE should be lowest of all categories. If not, the model is failing on 'easy' cases"
        filter:
          confidence_label: 1
        expected_range: [2, 10]
        count: 30

      - name: medium_confidence_cases
        description: "Mixed signals — a reasonable accuracy drop from high confidence"
        filter:
          confidence_label: 2
        expected_range: [1, 8]
        count: 30

      - name: low_confidence_cases
        description: "Cases you flagged as hard — rotation, injury, unknown factors"
        criteria: "High MAE expected. If model is confident here, it's overconfident"
        filter:
          confidence_label: 3
        expected_range: [0, 6]
        count: 30
```

### 4.3 Build the Eval Case Generator

Create `eval/build_cases.py` that reads the YAML and filters the test set to produce case files:

```python
import yaml
import pandas as pd
import os
import json

def build_eval_cases(features_path, suite_path, output_dir):
    """
    Reads eval_suite.yaml, filters the test set, and writes
    one JSONL file per case into eval/cases/
    """
    df = pd.read_csv(features_path)
    test = df[df["gameweek"] > 30]  # Same split as training

    with open(suite_path) as f:
        suite = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)
    case_manifest = []

    for category_name, category in suite["categories"].items():
        for case in category["cases"]:
            filtered = apply_filters(test, case.get("filter", {}))
            sampled = filtered.sample(
                n=min(case.get("count", 20), len(filtered)),
                random_state=42
            )

            case_file = f"{output_dir}/{case['name']}.jsonl"
            sampled.to_json(case_file, orient="records", lines=True)

            case_manifest.append({
                "name": case["name"],
                "category": category_name,
                "description": case.get("description", ""),
                "criteria": case.get("criteria", ""),
                "expected_range": case.get("expected_range", [0, 20]),
                "count": len(sampled),
                "file": case_file
            })

    # Save manifest for the runner
    with open(f"{output_dir}/manifest.json", "w") as f:
        json.dump(case_manifest, f, indent=2)

    print(f"Built {len(case_manifest)} eval cases")

def apply_filters(df, filters):
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        if key.endswith("_min"):
            col = key.replace("_min", "")
            mask &= df[col] >= value
        elif key.endswith("_max"):
            col = key.replace("_max", "")
            mask &= df[col] <= value
        else:
            mask &= df[key] == value
    return df[mask]
```

### 4.4 Output Quality Checks (LLM-Specific)

Create `eval/checks/output_checks.py` — validators that run on every LLM prediction:

```python
import re

def check_valid_integer(response: str) -> dict:
    """Did the model return a parseable integer?"""
    cleaned = response.strip()
    try:
        value = int(cleaned)
        return {"pass": True, "value": value, "check": "valid_integer"}
    except ValueError:
        # Fallback: extract first integer from response
        match = re.search(r'\b(\d+)\b', cleaned)
        if match:
            return {
                "pass": True,
                "value": int(match.group(1)),
                "check": "valid_integer",
                "warning": f"Extracted from noisy output: '{cleaned[:50]}'"
            }
        return {"pass": False, "value": None, "check": "valid_integer",
                "error": f"No integer found in: '{cleaned[:80]}'"}

def check_plausible_range(value: int) -> dict:
    """Is the prediction within a plausible FPL points range?"""
    # FPL points realistically range from -4 to ~25
    # (negative for red cards/own goals, max for hat-trick + bonus + clean sheet)
    if value is None:
        return {"pass": False, "check": "plausible_range"}
    in_range = -4 <= value <= 25
    return {
        "pass": in_range,
        "value": value,
        "check": "plausible_range",
        "error": None if in_range else f"Prediction {value} outside plausible range [-4, 25]"
    }

def check_expected_range(value: int, expected_min: int, expected_max: int) -> dict:
    """Is the prediction within the expected range for this eval case?"""
    if value is None:
        return {"pass": False, "check": "expected_range"}
    in_range = expected_min <= value <= expected_max
    return {
        "pass": in_range,
        "value": value,
        "check": "expected_range",
        "detail": f"Expected [{expected_min}, {expected_max}], got {value}"
    }

def run_all_checks(response: str, expected_range: list = None) -> dict:
    """Run all output checks and return a summary"""
    results = {}

    int_check = check_valid_integer(response)
    results["valid_integer"] = int_check

    if int_check["pass"]:
        results["plausible_range"] = check_plausible_range(int_check["value"])
        if expected_range:
            results["expected_range"] = check_expected_range(
                int_check["value"], expected_range[0], expected_range[1]
            )

    results["all_passed"] = all(r.get("pass", False) for r in results.values())
    results["parsed_value"] = int_check.get("value")

    return results
```

### 4.5 The Eval Runner

Create `eval/run_eval.py` — the single command that runs everything:

```bash
python eval/run_eval.py --models xgboost,llm,base_llm --tag v1-baseline
```

What it does:

1. Loads the case manifest
2. Runs each model against each eval case
3. For LLM outputs, runs all output quality checks
4. Computes per-case and per-category metrics
5. Compares against baseline (if one exists) and flags regressions
6. Saves timestamped results to `eval/results/{date}_{tag}/`
7. Prints a summary to terminal

```python
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from checks.output_checks import run_all_checks

def run_eval(models: dict, cases_dir: str, tag: str):
    """
    models: {"xgboost": predict_fn, "llm": predict_fn, "base_llm": predict_fn}
    Each predict_fn takes a DataFrame row and returns a prediction (int or str)
    """
    with open(f"{cases_dir}/manifest.json") as f:
        manifest = json.load(f)

    results = {"tag": tag, "timestamp": datetime.now().isoformat(), "models": {}}

    for model_name, predict_fn in models.items():
        model_results = {"categories": {}, "overall": {}, "checks": {}}
        all_preds = []
        all_actuals = []
        check_failures = 0
        total_checks = 0

        for case in manifest:
            case_df = pd.read_json(case["file"], lines=True)
            case_preds = []
            case_actuals = []

            for _, row in case_df.iterrows():
                raw_pred = predict_fn(row)
                actual = row["target_points"]

                # Run output checks for LLM models
                if model_name in ["llm", "base_llm"]:
                    checks = run_all_checks(str(raw_pred), case.get("expected_range"))
                    total_checks += 1
                    if not checks["all_passed"]:
                        check_failures += 1
                    pred = checks["parsed_value"] if checks["parsed_value"] is not None else 2
                else:
                    pred = float(raw_pred)

                case_preds.append(pred)
                case_actuals.append(actual)

            # Per-case metrics
            case_mae = mean_absolute_error(case_actuals, case_preds)
            expected_range = case.get("expected_range", [0, 20])
            in_range_pct = np.mean([
                expected_range[0] <= p <= expected_range[1] for p in case_preds
            ])

            category = case["category"]
            if category not in model_results["categories"]:
                model_results["categories"][category] = {"cases": {}}

            model_results["categories"][category]["cases"][case["name"]] = {
                "mae": round(case_mae, 3),
                "count": len(case_preds),
                "in_expected_range_pct": round(in_range_pct, 3),
                "description": case["description"]
            }

            all_preds.extend(case_preds)
            all_actuals.extend(case_actuals)

        # Category-level rollups
        for cat_name, cat_data in model_results["categories"].items():
            case_maes = [c["mae"] for c in cat_data["cases"].values()]
            cat_data["category_mae"] = round(np.mean(case_maes), 3)

        # Overall metrics
        model_results["overall"] = {
            "mae": round(mean_absolute_error(all_actuals, all_preds), 3),
            "rmse": round(np.sqrt(np.mean((np.array(all_actuals) - np.array(all_preds))**2)), 3),
            "within_1": round(np.mean(np.abs(np.array(all_actuals) - np.array(all_preds)) <= 1), 3),
            "within_3": round(np.mean(np.abs(np.array(all_actuals) - np.array(all_preds)) <= 3), 3),
        }

        # LLM-specific output quality
        if total_checks > 0:
            model_results["checks"] = {
                "total": total_checks,
                "failures": check_failures,
                "failure_rate": round(check_failures / total_checks, 3)
            }

        results["models"][model_name] = model_results

    return results
```

### 4.6 Regression Detection

Create `eval/compare.py` — compares a new eval run against the saved baseline:

```python
def compare_runs(new_results: dict, baseline_results: dict, threshold: float = 0.1):
    """
    Compare two eval runs. Flag regressions where MAE increased
    by more than threshold.
    """
    regressions = []
    improvements = []

    for model_name in new_results["models"]:
        if model_name not in baseline_results["models"]:
            continue

        new_model = new_results["models"][model_name]
        base_model = baseline_results["models"][model_name]

        # Overall regression check
        new_mae = new_model["overall"]["mae"]
        base_mae = base_model["overall"]["mae"]
        if new_mae > base_mae + threshold:
            regressions.append({
                "model": model_name,
                "level": "overall",
                "baseline_mae": base_mae,
                "new_mae": new_mae,
                "delta": round(new_mae - base_mae, 3)
            })
        elif new_mae < base_mae - threshold:
            improvements.append({
                "model": model_name,
                "level": "overall",
                "baseline_mae": base_mae,
                "new_mae": new_mae,
                "delta": round(new_mae - base_mae, 3)
            })

        # Per-category regression check
        for cat_name, cat_data in new_model["categories"].items():
            base_cat = base_model["categories"].get(cat_name, {})
            if not base_cat:
                continue

            new_cat_mae = cat_data["category_mae"]
            base_cat_mae = base_cat["category_mae"]

            if new_cat_mae > base_cat_mae + threshold:
                regressions.append({
                    "model": model_name,
                    "level": f"category:{cat_name}",
                    "baseline_mae": base_cat_mae,
                    "new_mae": new_cat_mae,
                    "delta": round(new_cat_mae - base_cat_mae, 3)
                })

        # Per-case regression check
        for cat_name, cat_data in new_model["categories"].items():
            base_cases = base_model["categories"].get(cat_name, {}).get("cases", {})
            for case_name, case_data in cat_data["cases"].items():
                base_case = base_cases.get(case_name, {})
                if not base_case:
                    continue
                if case_data["mae"] > base_case["mae"] + threshold:
                    regressions.append({
                        "model": model_name,
                        "level": f"case:{case_name}",
                        "baseline_mae": base_case["mae"],
                        "new_mae": case_data["mae"],
                        "delta": round(case_data["mae"] - base_case["mae"], 3)
                    })

    return {
        "regressions": regressions,
        "improvements": improvements,
        "has_regressions": len(regressions) > 0
    }
```

### 4.7 Eval Summary Output

After each run, `run_eval.py` generates `eval/results/{date}_{tag}/summary.md`:

```markdown
# Eval Run: v2-more-iters
**Date:** 2026-04-15
**Compared against:** v1-baseline

## Overall

| Model | MAE | RMSE | Within ±1 | Within ±3 | Output Failures |
|-------|-----|------|-----------|-----------|-----------------|
| XGBoost | 2.10 | 3.41 | 38.2% | 79.1% | — |
| Fine-Tuned LLM | 2.85 | 4.12 | 29.4% | 68.7% | 3.2% |
| Base LLM | 4.20 | 5.89 | 15.1% | 42.3% | 12.8% |

## Category Breakdown

| Category | XGBoost MAE | LLM MAE | Winner |
|----------|-------------|---------|--------|
| fixture_context | 2.05 | 2.72 | XGBoost |
| positional | 1.98 | 2.91 | XGBoost |
| edge_cases | 2.31 | 2.65 | XGBoost |
| high_value | 2.15 | 2.80 | XGBoost |
| human_confidence (high) | 1.62 | 2.10 | XGBoost |
| human_confidence (low) | 3.45 | 3.20 | LLM |

## Regressions vs Baseline (v1-baseline)

⚠️ 2 regressions detected:
- **LLM / case:player_on_hot_streak**: MAE 3.2 → 3.8 (+0.6)
- **LLM / category:edge_cases**: MAE 2.4 → 2.65 (+0.25)

✅ 1 improvement:
- **LLM / case:premium_mid_home_easy**: MAE 2.9 → 2.3 (−0.6)

## Output Quality (LLM only)

| Check | Pass Rate |
|-------|-----------|
| Valid integer returned | 96.8% |
| Within plausible range [-4, 25] | 98.2% |
| Within case expected range | 71.4% |
```

### 4.8 Running Evals in Practice

The workflow after every training experiment:

```bash
# 1. Fine-tune with new hyperparameters
mlx_lm.lora --model models/llama-3.2-3b --data data/mlx --train \
  --iters 1000 --learning-rate 5e-6 --adapter-path models/fpl-lora-adapter-v2

# 2. Run the eval suite (single command)
python eval/run_eval.py --models xgboost,llm,base_llm --tag v2-more-iters

# 3. Results appear immediately:
#    eval/results/2026-04-15_v2-more-iters/summary.md
#    eval/results/2026-04-15_v2-more-iters/scores.json
#    eval/results/2026-04-15_v2-more-iters/regressions.json

# 4. If happy with results, promote to baseline
cp eval/results/2026-04-15_v2-more-iters/scores.json eval/baseline/scores.json

# 5. Tag in git
git tag -a v2-more-iters -m "1000 iters, lr 5e-6 — improved fixture_context, regressed edge_cases"
```

### 4.9 Cost & Latency Benchmarking

This is what separates a technical experiment from an executive-level assessment. Measure the operational realities of each approach.

Create `eval/benchmark.py`:

```python
import time
import json
import psutil
import os

def benchmark_model(predict_fn, test_cases, model_name):
    """
    Measure wall-clock latency, memory, and cost for each model
    """
    latencies = []
    process = psutil.Process(os.getpid())

    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    for _, row in test_cases.iterrows():
        start = time.perf_counter()
        predict_fn(row)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    mem_after = process.memory_info().rss / 1024 / 1024

    return {
        "model": model_name,
        "avg_latency_ms": round(1000 * sum(latencies) / len(latencies), 1),
        "p95_latency_ms": round(1000 * sorted(latencies)[int(0.95 * len(latencies))], 1),
        "total_time_sec": round(sum(latencies), 1),
        "memory_mb": round(mem_after, 0),
        "predictions_per_second": round(len(latencies) / sum(latencies), 1),
    }
```

**What to record for each approach:**

| Metric | XGBoost | Few-Shot LLM | Fine-Tuned LLM | API (Claude/GPT) |
|--------|---------|-------------|-----------------|-------------------|
| Avg latency per prediction | ~1ms | ~2–5s | ~2–5s | ~1–3s |
| P95 latency | | | | |
| Memory footprint | ~50MB | ~2GB | ~2GB | 0 (cloud) |
| Cost per 1,000 predictions | ~$0 | ~$0 | ~$0 | ~$0.50–5.00 |
| Model size on disk | ~5MB | ~2GB | ~2GB + adapter | 0 |
| Setup/training time | ~2 min | 0 | ~20 min | 0 |
| Can run offline? | Yes | Yes | Yes | No |
| Can run on mobile/edge? | Yes | Maybe | Maybe | No |

**Optional but valuable:** Also run one test batch through an API model (Claude Sonnet via Anthropic API, or GPT-4o-mini) with the few-shot prompt. This gives you the cloud comparison data point — the cost difference between local and API at scale is one of the most useful numbers you can have.

Save results as `eval/benchmarks.json`.

### 4.10 Why This Matters for Product Work

Building this eval framework teaches you the things AI product leaders most commonly get wrong:

- **"Our model is 85% accurate" is meaningless** without knowing accurate at *what*. The category breakdown tells the real story.
- **Aggregate metrics hide regressions.** Your overall MAE can improve while a critical case (captain picks) gets worse. The regression detection catches this.
- **Output quality ≠ accuracy.** An LLM can have decent MAE but fail 12% of the time to even return a number. That's a product-breaking failure mode that MAE alone won't show you.
- **Eval cases are product requirements.** The `eval_suite.yaml` file is effectively a specification of what "good" looks like — the same skill as writing acceptance criteria.

---

## Phase 5: Comparison UI (Week 6)

### 5.1 Build the Streamlit Comparison UI

Create `ui/app.py`:

```python
import streamlit as st
import pandas as pd
import json

st.title("FPL Points Predictor: XGBoost vs Fine-Tuned LLM")

# Overall comparison table
comparison = pd.read_csv("eval/comparison.csv")
st.subheader("Overall Results")
st.dataframe(comparison)

# Per-player lookup
st.subheader("Player Lookup")
player = st.text_input("Enter player name")
gw = st.selectbox("Gameweek", range(31, 39))

if player:
    # Show XGBoost prediction, LLM prediction, actual
    # Show which model was closer
    pass

# Error distribution chart
st.subheader("Error Distribution")
# Histogram of (predicted - actual) for each model
```

Run with:
```bash
streamlit run ui/app.py
```

---

## Phase 6: Write-Up & Assessment (Week 7)

### 6.1 Document Your Findings

Create a write-up covering:

**1. Results Summary**
- Which model won on MAE? By how much?
- Which model was better at identifying high scorers?
- Did the LLM produce invalid outputs? How often?

**2. What Surprised You**
- Was the base LLM (zero-shot) any good at all?
- Did fine-tuning meaningfully improve the LLM over base?
- Were there player types where the LLM beat XGBoost?

**3. Product Leader Takeaways**

Answer these questions based on your experience:

| Question | Your answer from the project |
|----------|------------------------------|
| When should you fine-tune an LLM vs use traditional ML? | |
| When does prompting alone get you close enough? | |
| What's the real cost of data preparation for fine-tuning? | |
| How do you evaluate a model for a noisy prediction task? | |
| When does an LLM add value beyond raw accuracy? (explanation, flexibility, natural language interface) | |
| What's the minimum viable dataset size for useful fine-tuning? | |
| How much does local inference cost vs API inference? | |
| How did named eval cases change your understanding vs aggregate metrics alone? | |
| What did regression detection reveal that a single eval run wouldn't? | |
| How would you design evals for a production AI feature at your company? | |
| What did the manual labelling exercise teach you about data quality at scale? | |
| How did your human confidence labels correlate with model accuracy? | |
| Based on your results, would you ship this? In what form? | |
| What's the gap between "model works" and "product works"? | |

**4. If You Had More Time**
- Larger model (7B or 14B)
- Multi-season training data
- Ensemble approach (XGBoost prediction as a feature in the LLM prompt)
- Classification framing (predict "high return" vs "medium" vs "blank") instead of regression

### 6.2 Product Decision Framing

This is what separates an engineer's experiment from an executive's assessment. Take your results and answer: **"Would I ship this? And if so, how?"**

Write a one-page product brief (in `writeup/product_brief.md`) that covers:

**1. What would the user experience be?**

Don't just show a number. Design the output format based on what your models can actually deliver reliably:

| Model accuracy | Appropriate UX | Example |
|----------------|---------------|---------|
| MAE < 1.5 | Show exact prediction | "Salah: 8 pts" |
| MAE 1.5–3.0 | Show range | "Salah: 6–10 pts" |
| MAE > 3.0 | Show tier | "Salah: High return expected" |
| Low confidence cases | Show caveat | "Salah: 6–10 pts (rotation risk)" |

Based on your actual results, which UX is honest? If your model has MAE of 2.5, showing "8 points" is misleading. Showing "6–10 points" is more honest but less useful. This tension is at the heart of every AI product decision.

**2. What's the cost of being wrong?**

In FPL, a bad prediction might cost someone a captain pick (roughly 6–10 points swing in their team). That's annoying but not consequential. Map this to your platform:

- Low-stakes wrong: "Recommended player scored 2 instead of 7" → user is mildly annoyed
- Medium-stakes wrong: "Model says definitely captain Salah, he gets injured in warmup" → user loses trust
- High-stakes wrong: Consider domains where wrong predictions have real financial or safety consequences

Write one paragraph on how you'd set the accuracy threshold for shipping vs not shipping, and how that threshold would differ for different product contexts.

**3. Build vs buy vs prompt**

Based on your five-way comparison, fill in this decision framework:

```
For this prediction task, I would recommend:

□ Just prompt a frontier API model (Claude/GPT)
  → Because: few-shot gets within X% of fine-tuned, zero infra cost

□ Fine-tune an open-source model locally
  → Because: meaningful accuracy gain of X%, worth the training cost

□ Use traditional ML (XGBoost)
  → Because: best accuracy, fastest inference, simplest to maintain

□ Hybrid: XGBoost for prediction + LLM for explanation
  → Because: XGBoost wins on numbers but LLM can explain "why"

My reasoning: [2–3 sentences]
```

**4. The "explanation" advantage**

If you used chain-of-thought prompting in Phase 3, you got reasoning alongside predictions. Even if the LLM's accuracy was worse, can the explanations be a product feature? For example:

- XGBoost says: "Salah: 7 points" (no explanation)
- LLM says: "Salah: 8 points — Home fixture against a leaky defence (1.8 goals conceded per GW), strong recent form of 8.3, and Burnley have the worst away record. The only risk is Champions League rotation mid-week."

The second is a worse prediction but a better product. Write a short assessment of whether the explanation adds enough value to justify the accuracy trade-off.

**5. What would it take to go to production?**

List the gaps between your home project and a shippable product:

- Data freshness: How would you update predictions weekly? What's the pipeline?
- Monitoring: How would you know if the model starts degrading mid-season?
- Edge cases: What happens for new signings, double gameweeks, blank gameweeks?
- Scale: Your benchmarks show X ms per prediction — what's the cost at 10,000 users?
- Legal/ethical: Is there a gambling association risk? Would you need disclaimers?

This section demonstrates that you think beyond "does the model work" to "can we ship this responsibly."

---

## Phase 7: GitHub Repository & Notebooks (Week 8–9)

### 7.1 Create the Repository

```bash
cd fpl-predictor
git init
git remote add origin git@github.com:YOUR_USERNAME/fpl-llm-vs-xgboost.git
```

Create a `.gitignore` before your first commit:

```
# Models (too large for GitHub)
models/llama-3.2-3b/
models/fpl-lora-adapter/
models/xgboost_fpl.json

# Raw data (re-fetchable from API)
data/raw/

# Python
.venv/
__pycache__/
*.pyc

# OS
.DS_Store
```

**Important:** Model weights and raw API data stay local. Everything else — code, notebooks, processed data samples, eval results, charts — goes into the repo.

### 7.2 Jupyter Notebooks

Install Jupyter:

```bash
pip install jupyter nbformat
```

Create one notebook per phase in `/notebooks/`. Each notebook should mix code, output, charts, and your commentary explaining what you're seeing and why it matters.

**Notebook 1: `01_data_exploration.ipynb`**

Cover:
- How many players, gameweeks, total data points
- Distribution of points scored (histogram — show how skewed it is toward 2-3 points)
- Points by position (box plot — show that midfielders have highest variance)
- Feature correlation heatmap — which features correlate most with points?
- Sample rows from the final feature set
- Commentary: what makes this prediction problem hard (noise, rare events, positional differences)

**Notebook 2: `02_xgboost_training.ipynb`**

Cover:
- Training code with inline results
- Learning curves (training loss vs validation loss over iterations)
- Feature importance chart with your interpretation
- Prediction vs actual scatter plot
- Residual analysis — where does the model fail? (show it misses hat tricks, red cards)
- Commentary: what does feature importance tell you about FPL?

**Notebook 3: `03_llm_finetuning.ipynb`**

Cover:
- Example training prompts (show 3–4 to illustrate the format)
- Training loss curve from MLX
- Sample predictions from base model vs fine-tuned model (side by side)
- Failure analysis — show examples where the LLM returned garbage, and how you handled it
- Commentary: what did fine-tuning actually teach the model? Where does it still hallucinate?

**Notebook 4: `04_eval_framework.ipynb`**

Cover:
- Walk through the eval_suite.yaml — explain why you chose each category and case
- Show the output quality check results: what percentage of LLM responses were valid integers? Plausible range?
- Per-category MAE breakdown visualised as a grouped bar chart (XGBoost vs LLM per category)
- Regression detection output from a real experiment — show what it looks like when a training change improves one category but degrades another
- Show 5–10 specific eval case predictions side by side with commentary: "The LLM predicted 3, XGBoost predicted 5, actual was 2 — here's why"
- Commentary: what you learned about building evals, and how this applies to AI products in general

**Notebook 5: `05_comparison_and_assessment.ipynb`**

Cover:
- Full comparison table (all metrics, all three models)
- Error distribution histograms overlaid
- Position-by-position MAE bar chart
- High-scorer analysis — scatter plot for 8+ point players
- Head-to-head: for each test prediction, which model was closer? (pie chart or bar)
- Your written assessment and product takeaways
- This notebook IS the portfolio piece — spend time making it clear and polished

### 7.3 README.md Template

Create a README that works as a standalone summary for anyone visiting the repo:

```markdown
# FPL Points Prediction: Fine-Tuned LLM vs XGBoost

Can a locally fine-tuned open-source LLM compete with XGBoost at predicting
Fantasy Premier League player points?

## TL;DR Results

| Model | MAE | RMSE | Within ±1 pt | Within ±3 pts | Correlation |
|-------|-----|------|-------------|---------------|-------------|
| XGBoost | X.XX | X.XX | XX% | XX% | 0.XX |
| Fine-Tuned Llama 3.2 3B | X.XX | X.XX | XX% | XX% | 0.XX |
| Base Llama 3.2 3B (no fine-tuning) | X.XX | X.XX | XX% | XX% | 0.XX |

> **Key finding:** [One sentence summary — e.g. "XGBoost wins on accuracy
> but the fine-tuned LLM produces surprisingly coherent reasoning about
> fixture difficulty."]

## What This Project Is

A hands-on comparison of two AI approaches to the same prediction problem,
built as a learning project to understand when LLM fine-tuning adds value
vs traditional ML.

**Built with:**
- MLX + LoRA fine-tuning on Mac Mini M4 (no cloud GPU needed)
- Llama 3.2 3B (4-bit quantised)
- XGBoost for the ML baseline
- FPL public API for all data

## Project Structure

```
notebooks/           Documented experiments (start here)
  01_data_exploration.ipynb
  02_xgboost_training.ipynb
  03_llm_finetuning.ipynb
  04_eval_framework.ipynb
  05_comparison_and_assessment.ipynb
data/                Pipeline scripts + processed data
models/              Training scripts (weights not included)
eval/                Eval suite, case definitions, results history
ui/                  Streamlit comparison dashboard
```

## Key Takeaways

### When to fine-tune an LLM vs use traditional ML
[Your answer]

### What surprised me
[Your answer]

### The real cost of data preparation
[Your answer]

## Reproduce This

```bash
git clone https://github.com/YOUR_USERNAME/fpl-llm-vs-xgboost.git
cd fpl-llm-vs-xgboost
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fetch data
python data/fetch_fpl.py
python data/build_features.py

# Train XGBoost
python models/train_xgboost.py

# Fine-tune LLM (requires Mac with Apple Silicon)
pip install mlx mlx-lm
huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit --local-dir models/llama-3.2-3b
python data/build_mlx_prompts.py
mlx_lm.lora --model models/llama-3.2-3b --data data/mlx --train --iters 600 --adapter-path models/fpl-lora-adapter

# Evaluate
python eval/evaluate.py

# Dashboard
streamlit run ui/app.py
```

## Hardware

- Mac Mini M4 (16GB unified memory)
- No cloud GPU required
- Fine-tuning takes ~20 minutes, inference ~5 minutes for full test set
```

### 7.4 Experiment Tracking with Git Tags

Tag each meaningful experiment so you can compare and roll back:

```bash
# After first successful fine-tune
git tag -a v1-baseline -m "First fine-tune: 600 iters, lr 1e-5, 8 lora layers"

# After tuning hyperparameters
git tag -a v2-more-iters -m "1000 iters, lr 5e-6, 16 lora layers"

# After trying a different base model
git tag -a v3-mistral-7b -m "Switched to Mistral 7B base model"
```

Keep a simple experiment log in `eval/experiment_log.md`:

```markdown
# Experiment Log

| Tag | Base Model | Iters | LR | LoRA Layers | MAE | Notes |
|-----|-----------|-------|-----|-------------|-----|-------|
| v1-baseline | Llama 3.2 3B | 600 | 1e-5 | 8 | X.XX | First run |
| v2-more-iters | Llama 3.2 3B | 1000 | 5e-6 | 16 | X.XX | Slight improvement |
| v3-mistral-7b | Mistral 7B | 600 | 1e-5 | 8 | X.XX | Better on high scorers |
```

### 7.5 GitHub Actions (Optional but Impressive)

Add `.github/workflows/eval.yml` to auto-run evals when you push changes:

```yaml
name: Run Evaluation

on:
  push:
    paths:
      - 'eval/**'
      - 'models/train_xgboost.py'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python eval/evaluate.py
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: eval/comparison.csv
```

Note: This only works for the XGBoost eval (GitHub Actions runners don't have Apple Silicon for MLX). But it proves the XGBoost pipeline is reproducible.

### 7.6 Publishing the Write-Up (GitHub Pages)

Turn your final notebook into a published page:

```bash
pip install jupyter nbconvert

# Convert the comparison notebook to HTML
jupyter nbconvert --to html notebooks/05_comparison_and_assessment.ipynb --output-dir docs/

# Enable GitHub Pages in repo settings → Source: /docs folder
```

This gives you a public URL like `https://YOUR_USERNAME.github.io/fpl-llm-vs-xgboost/` that you can share.

---

## File Structure

```
fpl-predictor/
├── .github/
│   └── workflows/
│       └── eval.yml                # Auto-run XGBoost eval on push
├── .gitignore
├── data/
│   ├── fetch_fpl.py              # API fetcher
│   ├── build_features.py         # Feature engineering → CSV
│   ├── build_prompts.py          # Prompt engineering → JSONL
│   ├── build_mlx_prompts.py      # MLX chat format → JSONL
│   ├── raw/                      # Raw API responses (gitignored)
│   │   ├── players.json
│   │   ├── fixtures.json
│   │   └── player_histories/
│   ├── processed/
│   │   ├── features.csv
│   │   ├── train_prompts.jsonl
│   │   └── test_prompts.jsonl
│   ├── labelled/                  # Manual confidence labels (100 rows)
│   │   └── confidence_labels.csv
│   └── mlx/                      # MLX chat format splits
│       ├── train.jsonl
│       ├── valid.jsonl
│       └── test.jsonl
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_xgboost_training.ipynb
│   ├── 03_llm_finetuning.ipynb
│   ├── 04_eval_framework.ipynb
│   └── 05_comparison_and_assessment.ipynb
├── models/
│   ├── train_xgboost.py
│   ├── predict_llm.py
│   ├── llama-3.2-3b/             # Downloaded base model (gitignored)
│   ├── fpl-lora-adapter/         # Fine-tuned adapter weights (gitignored)
│   └── xgboost_fpl.json          # Trained XGBoost model (gitignored)
├── eval/
│   ├── eval_suite.yaml            # Defines all eval cases and categories
│   ├── run_eval.py                # Single command eval runner
│   ├── compare.py                 # Regression detection between runs
│   ├── build_cases.py             # Generates eval case files from YAML + test data
│   ├── cases/                     # Named eval scenario data files
│   │   ├── manifest.json
│   │   ├── premium_mid_home_easy.jsonl
│   │   ├── captain_pick_candidates.jsonl
│   │   └── ...
│   ├── checks/                    # LLM output quality validators
│   │   └── output_checks.py
│   ├── results/                   # Timestamped results from every run
│   │   └── {date}_{tag}/
│   │       ├── scores.json
│   │       ├── regressions.json
│   │       └── summary.md
│   ├── baseline/                  # Current best run to compare against
│   │   └── scores.json
│   └── experiment_log.md          # Tracks every fine-tuning run
├── ui/
│   └── app.py                    # Streamlit comparison dashboard
├── docs/                          # GitHub Pages output
│   └── 05_comparison_and_assessment.html
├── writeup/
│   ├── assessment.md             # Final write-up
│   └── product_brief.md          # "Would I ship this?" product decision
├── requirements.txt
└── README.md                      # Project landing page with results
```

---

## Tools & Dependencies

| Tool | Purpose | Install |
|------|---------|---------|
| Python 3.11+ | Everything | Already on Mac Mini |
| mlx + mlx-lm | Local LLM fine-tuning & inference | `pip install mlx mlx-lm` |
| xgboost | Baseline ML model | `pip install xgboost` |
| scikit-learn | Metrics & utilities | `pip install scikit-learn` |
| pandas | Data manipulation | `pip install pandas` |
| streamlit | Comparison UI | `pip install streamlit` |
| huggingface_hub | Model downloads | `pip install huggingface_hub` |
| matplotlib | Charts | `pip install matplotlib` |
| jupyter | Notebooks for documented experiments | `pip install jupyter nbconvert` |
| pyyaml | Eval suite configuration | `pip install pyyaml` |

**Mac Mini M4 RAM requirements:**
- Llama 3.2 3B (4-bit): ~2GB during inference, ~4GB during fine-tuning
- Mistral 7B (4-bit): ~4GB during inference, ~8GB during fine-tuning
- XGBoost: negligible

---

## Claude Code Usage Guide

Use Claude Code at each phase for:

| Phase | Claude Code tasks |
|-------|-------------------|
| Data Pipeline | Write the API fetcher, feature engineering script, handle edge cases |
| Data Labelling | Generate the 100-row sample, build the labelling spreadsheet template |
| XGBoost | Hyperparameter tuning script, feature importance analysis |
| Prompt Engineering | Build few-shot and chain-of-thought prompt templates, batch runner |
| LLM Fine-Tune | Debug MLX config, write batch inference script, build output parser |
| Eval Framework | Build the eval suite YAML, case generator, output checks, regression detector |
| Comparison UI | Scaffold the Streamlit app, add interactivity |
| Write-up & Product Brief | Review findings, structure the assessment, draft the product brief |
| GitHub & Notebooks | Generate notebook scaffolds, write README, set up GitHub Actions workflow |

**Tip:** Paste error messages directly into Claude Code. MLX errors in particular can be cryptic — Claude Code has seen most of them.
