<p align="center">
  <img src="assets/hero-banner.svg" alt="FPL Points Predictor — Fine-Tuned LLM vs XGBoost" width="100%">
</p>

<p align="center">
  <strong>Can a locally fine-tuned open-source LLM compete with XGBoost at predicting Fantasy Premier League player points?</strong>
</p>

<p align="center">
  <a href="#the-experiment">The Experiment</a> ·
  <a href="#how-it-works">How It Works</a> ·
  <a href="#project-structure">Project Structure</a> ·
  <a href="#current-progress">Progress</a> ·
  <a href="#reproduce-this">Reproduce</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-work%20in%20progress-orange?style=flat-square" alt="Status: WIP">
  <img src="https://img.shields.io/badge/python-3.11-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Apple%20Silicon-MLX-black?style=flat-square&logo=apple&logoColor=white" alt="Apple Silicon MLX">
  <img src="https://img.shields.io/badge/model-Llama%203.2%203B-purple?style=flat-square" alt="Llama 3.2 3B">
</p>

---

> **This project is actively under construction.** Results, notebooks, and the comparison UI are not yet complete. Check back for updates or watch the repo.

---

## The Experiment

Two AI approaches. Same prediction problem. One winner.

**Approach A — XGBoost (Traditional ML):** Feed structured player stats into a gradient-boosted decision tree. Fast, proven, and the industry standard for tabular prediction tasks.

**Approach B — Fine-Tuned LLM (Llama 3.2 3B):** Take an open-source language model, fine-tune it with LoRA on natural language descriptions of player stats, and ask it to predict points. Runs entirely on a Mac Mini M4 using Apple's MLX framework — no cloud GPU needed.

The question isn't just "which is more accurate?" It's: **when does an LLM add value over traditional ML, and what are the real-world trade-offs?**

## How It Works

### Data Pipeline

All data comes from the free, public [FPL API](https://fantasy.premierleague.com/api/bootstrap-static/). The pipeline fetches player stats, fixture data, and gameweek histories for 800+ players, then engineers 20 features per player-gameweek:

| Feature Type | Examples |
|:---|:---|
| **Rolling form** | Points avg (3/5 GW), minutes avg, bonus avg, ICT index avg |
| **Season stats** | Goals per 90, assists per 90, clean sheet % |
| **Fixture context** | Home/away, fixture difficulty rating (1-5) |
| **Team context** | Team goals scored/conceded (last 3 GW), opponent goals conceded (season avg) |

All rolling features are computed using only **prior gameweeks** — no future data leakage.

### The Models

```
                              ┌─────────────────────────────┐
                              │   8,468 player-gameweek     │
                              │   training examples         │
                              └──────────┬──────────────────┘
                                         │
                          ┌──────────────┴──────────────┐
                          ▼                             ▼
                 ┌─────────────────┐          ┌─────────────────┐
                 │    XGBoost      │          │  Llama 3.2 3B   │
                 │                 │          │  + LoRA adapter  │
                 │  Tabular CSV    │          │                 │
                 │  → Regression   │          │  Chat prompts   │
                 │                 │          │  → Integer pred  │
                 └────────┬────────┘          └────────┬────────┘
                          │                            │
                          └──────────┬─────────────────┘
                                     ▼
                          ┌─────────────────────┐
                          │  Evaluation Harness  │
                          │  MAE, RMSE, ±1/±3   │
                          │  Category breakdown  │
                          │  Regression detect   │
                          └─────────────────────┘
```

### Evaluation Framework

Not just "what's the MAE?" — a structured eval suite with named test scenarios:

| Category | What it tests |
|:---|:---|
| **Fixture context** | Does the model understand home vs away, easy vs hard fixtures? |
| **Positional** | Can it handle GK/DEF/MID/FWD differently? |
| **Edge cases** | Low-minutes players, hot streaks, weak teams |
| **High value** | Captain picks, differentials — the predictions that matter |
| **Human confidence** | Accuracy on cases a human labelled as easy vs hard to predict |

Every experiment run is compared against a saved baseline, with automatic regression detection.

## Results

> **Coming soon.** Model training and evaluation are in progress.

| Model | MAE | RMSE | Within ±1 pt | Within ±3 pts |
|:---|:---|:---|:---|:---|
| XGBoost | _pending_ | _pending_ | _pending_ | _pending_ |
| Fine-Tuned LLM | _pending_ | _pending_ | _pending_ | _pending_ |
| Base LLM (no fine-tuning) | _pending_ | _pending_ | _pending_ | _pending_ |
| Few-Shot LLM | _pending_ | _pending_ | _pending_ | _pending_ |
| Chain-of-Thought LLM | _pending_ | _pending_ | _pending_ | _pending_ |

## Current Progress

- [x] **Phase 1: Data Pipeline** — FPL API fetcher, feature engineering (20 features), prompt generation
- [x] **Phase 2: XGBoost Baseline** — Trained and evaluated (MAE 2.11)
- [x] **Phase 3: LLM Fine-Tuning** — LoRA fine-tuned Llama 3.2 3B with MLX (MAE 1.96)
- [ ] **Phase 4: Eval Framework** — Category-based eval suite, output quality checks, regression detection
- [x] **Phase 5: Comparison Dashboard** — Multi-page Streamlit app with predictions table, fixture lookahead, squad builder, and transfer advisor
- [ ] **Phase 6: Write-Up** — Product assessment and "would I ship this?" analysis
- [ ] **Phase 7: Notebooks & Docs** — Documented experiments and findings

## Project Structure

```
data/
  fetch_fpl.py              Fetches all data from FPL API
  build_features.py         Engineers 20 features per player-gameweek
  build_prompts.py          Generates LLM training prompts (JSONL + MLX chat format)
  processed/                Feature CSV + prediction outputs
  mlx/                      MLX-formatted train/valid/test splits
  raw/                      Raw JSON from FPL API (players, fixtures, teams, gameweeks)
models/
  train_xgboost.py          XGBoost training script
  predict_next_gw.py        Next-GW prediction pipeline (XGBoost + LLM)
  predict_llm.py            LLM evaluation harness (4 strategies)
  xgboost_fpl.json          Pre-trained XGBoost model
  llama-3.2-3b/             Base model (4-bit quantised)
  fpl-lora-adapter-v2/      Fine-tuned LoRA weights
eval/
  eval_suite.yaml           Named eval scenarios and categories
  run_eval.py               Single-command eval runner
  compare.py                Regression detection between runs
ui/
  app.py                    Entry point — multi-page navigation shell
  data_loader.py            Cached data loading + fixture lookahead
  styles.py                 FPL-themed CSS styling
  components.py             Reusable HTML components (badges, cards, etc.)
  pages/
    1_Predictions.py        Main predictions table with filters + next 2 fixtures
    2_My_Team.py            Squad builder + multi-transfer advisor
```

## Reproduce This

```bash
git clone https://github.com/tom-barkan/FPL-MLprediction-model.git
cd FPL-MLprediction-model

# Set up environment
brew install python@3.11 libomp
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fetch data from FPL API (~15 min due to rate limiting)
python data/fetch_fpl.py

# Build features and prompts
python data/build_features.py
python data/build_prompts.py

# Train XGBoost
python models/train_xgboost.py

# Fine-tune LLM (requires Apple Silicon)
pip install mlx mlx-lm
huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit --local-dir models/llama-3.2-3b
mlx_lm.lora --model models/llama-3.2-3b --data data/mlx --train --iters 600 --adapter-path models/fpl-lora-adapter-v2

# Generate next-GW predictions (both models)
python models/predict_next_gw.py

# Launch the dashboard
streamlit run ui/app.py
```

### Dashboard

The Streamlit dashboard has two pages:

**Predictions** — The main page with a sortable/filterable table showing XGBoost and LLM predictions for all 344 players. Includes next 2 fixtures with fixture difficulty ratings, confidence progress bars, player deep dive, and model agreement analysis.

**My Team** — Build your 15-player FPL squad and get transfer recommendations. Features squad validation (position limits, max 3 per team, 100m budget), auto-picked starting XI, and a multi-transfer planner with four strategy tabs:
- **Safe** — High confidence from both models
- **Differential** — High points, lower confidence (risk/reward)
- **Form** — Players trending up in recent gameweeks
- **Fixture** — Easiest upcoming fixtures

## Hardware

| Component | Spec |
|:---|:---|
| Machine | Mac Mini M4 |
| RAM | 16GB unified memory |
| Cloud GPU | Not required |
| Fine-tuning time | ~20 minutes |
| Inference (full test set) | ~5 minutes |

## Tech Stack

| Tool | Purpose |
|:---|:---|
| **MLX + mlx-lm** | Local LLM fine-tuning and inference on Apple Silicon |
| **Llama 3.2 3B** | Base model for fine-tuning (4-bit quantised, ~2GB) |
| **XGBoost** | Traditional ML baseline |
| **FPL API** | Free, public source for all player and fixture data |
| **Streamlit** | Comparison dashboard |
| **Jupyter** | Documented experiment notebooks |

## License

[MIT](LICENSE)
