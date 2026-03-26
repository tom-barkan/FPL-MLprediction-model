"""
Run LLM inference on the test set using multiple strategies:
  1. Base model (zero-shot)
  2. Base model (few-shot)
  3. Base model (chain-of-thought)
  4. Fine-tuned model (with LoRA adapter)

Parses integer predictions from LLM output and saves results.
"""

import json
import os
import re
import time
import numpy as np
import pandas as pd
from mlx_lm import load, generate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "llama-3.2-3b")
ADAPTER_DIR = os.path.join(ROOT, "models", "fpl-lora-adapter-v3")  # v3: per-point balanced, 16 layers, 500 iters
MLX_DIR = os.path.join(ROOT, "data", "mlx")
EVAL_DIR = os.path.join(ROOT, "eval")

SYSTEM_PROMPT = (
    "You are an FPL points predictor. Given player stats and fixture info, "
    "predict the total FPL points for the gameweek. Respond with only a single integer."
)

COT_SYSTEM_PROMPT = (
    "You are an FPL points predictor. Think step by step about the player's "
    "form, fixture difficulty, and position before giving your prediction. "
    'End your response with "Prediction: X" where X is a single integer.'
)

# Few-shot examples (drawn from training data, covering different positions/outcomes)
FEW_SHOT_EXAMPLES = [
    {
        "user": (
            "Player: Salah (MID)\nGameweek: 15\nFixture: Home (FDR: 2)\n"
            "Form (last 3 GW avg): 8.3 pts\nForm (last 5 GW avg): 7.2 pts\n"
            "Minutes (last 3 GW avg): 90.0\nGoals per 90 (season): 0.62\n"
            "Assists per 90 (season): 0.31\nICT index (last 3 GW avg): 12.1\n"
            "Clean sheet % (season): 0.0%\nBonus avg (last 3 GW): 1.7\n"
            "Team goals scored (last 3 GW): 9\nTeam goals conceded (last 3 GW): 2\n"
            "Opponent goals conceded (season avg per GW): 1.8\nPrice: 13.2"
        ),
        "assistant": "8",
    },
    {
        "user": (
            "Player: Gabriel (DEF)\nGameweek: 18\nFixture: Away (FDR: 5)\n"
            "Form (last 3 GW avg): 3.0 pts\nForm (last 5 GW avg): 3.4 pts\n"
            "Minutes (last 3 GW avg): 90.0\nGoals per 90 (season): 0.08\n"
            "Assists per 90 (season): 0.04\nICT index (last 3 GW avg): 3.2\n"
            "Clean sheet % (season): 41.2%\nBonus avg (last 3 GW): 0.3\n"
            "Team goals scored (last 3 GW): 4\nTeam goals conceded (last 3 GW): 5\n"
            "Opponent goals conceded (season avg per GW): 0.8\nPrice: 6.2"
        ),
        "assistant": "2",
    },
    {
        "user": (
            "Player: Watkins (FWD)\nGameweek: 12\nFixture: Home (FDR: 3)\n"
            "Form (last 3 GW avg): 4.7 pts\nForm (last 5 GW avg): 4.2 pts\n"
            "Minutes (last 3 GW avg): 82.0\nGoals per 90 (season): 0.35\n"
            "Assists per 90 (season): 0.12\nICT index (last 3 GW avg): 6.8\n"
            "Clean sheet % (season): 0.0%\nBonus avg (last 3 GW): 0.3\n"
            "Team goals scored (last 3 GW): 5\nTeam goals conceded (last 3 GW): 4\n"
            "Opponent goals conceded (season avg per GW): 1.4\nPrice: 7.8"
        ),
        "assistant": "4",
    },
]


def parse_prediction(response):
    """Extract an integer prediction from LLM output."""
    cleaned = response.strip()

    # Try direct integer parse
    try:
        return int(cleaned)
    except ValueError:
        pass

    # Try "Prediction: X" pattern (for CoT)
    match = re.search(r'[Pp]rediction:\s*(\d+)', cleaned)
    if match:
        return int(match.group(1))

    # Fall back to first integer in response
    match = re.search(r'\b(\d+)\b', cleaned)
    if match:
        return int(match.group(1))

    return None


def build_chat_prompt(tokenizer, messages):
    """Apply the model's chat template to a list of messages."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_inference(model, tokenizer, test_data, strategy="zero_shot", max_tokens=20):
    """Run inference on test data and return predictions."""
    results = []

    for i, item in enumerate(test_data):
        user_content = item["messages"][1]["content"]
        actual = int(item["messages"][2]["content"])

        # Build messages based on strategy
        if strategy == "zero_shot":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        elif strategy == "few_shot":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for ex in FEW_SHOT_EXAMPLES:
                messages.append({"role": "user", "content": ex["user"]})
                messages.append({"role": "assistant", "content": ex["assistant"]})
            messages.append({"role": "user", "content": user_content})
        elif strategy == "chain_of_thought":
            messages = [
                {"role": "system", "content": COT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            max_tokens = 150  # CoT needs more tokens
        elif strategy == "fine_tuned":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        prompt = build_chat_prompt(tokenizer, messages)
        response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)

        parsed = parse_prediction(response)

        results.append({
            "index": i,
            "actual": actual,
            "raw_response": response[:200],
            "parsed_prediction": parsed,
            "parse_success": parsed is not None,
        })

        if (i + 1) % 25 == 0:
            successes = sum(1 for r in results if r["parse_success"])
            print(f"  [{strategy}] {i+1}/{len(test_data)} — parse rate: {successes}/{len(results)}")

    return results


def evaluate_results(results, strategy_name):
    """Compute metrics from inference results."""
    valid = [r for r in results if r["parse_success"]]
    if not valid:
        return {"strategy": strategy_name, "mae": None, "parse_failure_rate": 1.0}

    actuals = np.array([r["actual"] for r in valid])
    preds = np.array([r["parsed_prediction"] for r in valid])
    errors = np.abs(actuals - preds)

    return {
        "strategy": strategy_name,
        "mae": round(float(np.mean(errors)), 4),
        "rmse": round(float(np.sqrt(np.mean((actuals - preds) ** 2))), 4),
        "within_1": round(float(np.mean(errors <= 1)), 4),
        "within_3": round(float(np.mean(errors <= 3)), 4),
        "parse_failure_rate": round(1 - len(valid) / len(results), 4),
        "total_predictions": len(results),
        "valid_predictions": len(valid),
    }


def load_test_data():
    """Load the MLX-format test set."""
    test_path = os.path.join(MLX_DIR, "test.jsonl")
    data = []
    with open(test_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    test_data = load_test_data()
    print(f"Loaded {len(test_data)} test cases\n")

    all_results = {}

    # --- Base model strategies ---
    print("Loading base model...")
    base_model, base_tokenizer = load(MODEL_DIR)

    for strategy in ["zero_shot", "few_shot", "chain_of_thought"]:
        print(f"\nRunning {strategy}...")
        start = time.time()
        raw_results = run_inference(base_model, base_tokenizer, test_data, strategy=strategy)
        elapsed = time.time() - start

        metrics = evaluate_results(raw_results, strategy)
        metrics["time_seconds"] = round(elapsed, 1)
        all_results[strategy] = metrics

        # Save raw predictions
        with open(os.path.join(EVAL_DIR, f"llm_{strategy}_predictions.json"), "w") as f:
            json.dump(raw_results, f, indent=2)

        print(f"  {strategy}: MAE={metrics['mae']}, parse_fail={metrics['parse_failure_rate']:.1%}, time={elapsed:.0f}s")

    # Free base model memory
    del base_model, base_tokenizer

    # --- Fine-tuned model ---
    if os.path.exists(ADAPTER_DIR):
        print(f"\nLoading fine-tuned model (adapter: {ADAPTER_DIR})...")
        ft_model, ft_tokenizer = load(MODEL_DIR, adapter_path=ADAPTER_DIR)

        print("Running fine_tuned...")
        start = time.time()
        raw_results = run_inference(ft_model, ft_tokenizer, test_data, strategy="fine_tuned")
        elapsed = time.time() - start

        metrics = evaluate_results(raw_results, "fine_tuned")
        metrics["time_seconds"] = round(elapsed, 1)
        all_results["fine_tuned"] = metrics

        with open(os.path.join(EVAL_DIR, "llm_fine_tuned_predictions.json"), "w") as f:
            json.dump(raw_results, f, indent=2)

        print(f"  fine_tuned: MAE={metrics['mae']}, parse_fail={metrics['parse_failure_rate']:.1%}, time={elapsed:.0f}s")

        del ft_model, ft_tokenizer
    else:
        print(f"\nSkipping fine-tuned model — adapter not found at {ADAPTER_DIR}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("LLM RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Strategy':<20} {'MAE':>8} {'RMSE':>8} {'±1':>8} {'±3':>8} {'Parse%':>8}")
    print(f"{'-'*60}")
    for name, m in all_results.items():
        mae = f"{m['mae']:.2f}" if m['mae'] is not None else "N/A"
        rmse = f"{m.get('rmse', 0):.2f}" if m.get('rmse') else "N/A"
        w1 = f"{m.get('within_1', 0):.1%}" if m.get('within_1') else "N/A"
        w3 = f"{m.get('within_3', 0):.1%}" if m.get('within_3') else "N/A"
        parse = f"{1-m['parse_failure_rate']:.1%}"
        print(f"{name:<20} {mae:>8} {rmse:>8} {w1:>8} {w3:>8} {parse:>8}")

    # Save summary
    with open(os.path.join(EVAL_DIR, "llm_results_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {EVAL_DIR}/")


if __name__ == "__main__":
    main()
