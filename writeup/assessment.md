# FPL Prediction Assessment: What We Built, What We Learned

## 1. Results Summary

### The Scoreboard

| Model | MAE | RMSE | Within ±1 pt | Within ±3 pts | Inference Time |
|:------|----:|-----:|-------------:|:-------------:|:--------------:|
| **XGBoost** | **2.11** | **2.81** | 26.7% | **82.3%** | <1s |
| Few-Shot LLM | 2.16 | 3.32 | 55.6% | 76.1% | 370s |
| Chain-of-Thought LLM | 2.20 | 3.60 | **60.1%** | 73.7% | 845s |
| Fine-Tuned LLM v3 | 3.35 | 4.46 | 39.1% | 65.8% | 148s |
| Zero-Shot LLM | 11.54 | 14.22 | 2.1% | 10.3% | 133s |

**XGBoost wins on raw accuracy.** Lowest MAE (2.11), lowest RMSE (2.81), and highest ±3 accuracy (82.3%). It's the clear winner if your only metric is "how close are the predictions to reality."

### Category Breakdown (from eval suite)

The category-level eval reveals where each model has strengths and weaknesses:

| Category | XGBoost | Few-Shot | CoT | Fine-Tuned v3 |
|:---------|--------:|---------:|----:|--------------:|
| Home fixtures | 2.19 | 2.22 | 2.38 | 3.47 |
| Positional (GK home) | 1.58 | 2.25 | 2.88 | 2.38 |
| Budget DEF away | 2.25 | 1.93 | 2.11 | 2.15 |
| Low minutes (edge) | 1.95 | 1.77 | 1.63 | 2.95 |
| Cold players | 1.65 | 1.42 | 1.34 | 2.22 |
| Hot streak | 3.29 | 3.75 | 4.00 | 6.25 |
| Captain picks | 4.28 | 5.00 | 4.33 | 7.00 |

**Key findings from the category breakdown:**
- **XGBoost is best at goalkeepers** (MAE 1.58 for GK at home) — positional expertise from structured features
- **Few-shot and CoT beat XGBoost on cold/low-form players** — the LLM is better at predicting "nothing will happen"
- **All models struggle with hot streaks and captain picks** — the highest-value predictions are the hardest
- **Fine-tuned v3 is worst at captain picks** (MAE 7.00) — its wide range means bigger misses on premium players

### What Surprised Us

1. **Few-shot nearly matches XGBoost.** MAE 2.16 vs 2.11 — a 0.05 difference with zero training, zero infrastructure. This is the single most important finding for the "build vs prompt" decision.

2. **Chain-of-thought beats few-shot on ±1 accuracy (60.1% vs 55.6%).** The reasoning step helps calibration — CoT is more precise when it's right, but makes occasional larger errors (higher RMSE).

3. **Zero-shot is completely useless (MAE 11.54).** The base model has no innate knowledge of FPL scoring. You cannot just "ask ChatGPT" — you need examples or training.

4. **Fine-tuning made accuracy worse, not better.** The v3 fine-tuned model has MAE 3.35 — worse than both few-shot (2.16) and CoT (2.20). The trade-off was intentional: wider prediction range for better player *ranking*, at the cost of worse point *estimation*.

5. **The real problem was data quality, not model capacity.** Three iterations of fine-tuning taught us that fixing the training data distribution (mode collapse) mattered far more than scaling LoRA layers or tuning hyperparameters.

---

## 2. Product Leader Takeaways

### When should you fine-tune an LLM vs use traditional ML?

**Fine-tune when:** You need the model to behave in a way that few-shot prompting can't replicate — e.g., consistent output format, domain-specific reasoning, or predictions across a range that the base model doesn't naturally produce.

**Use traditional ML when:** Your data is tabular, your features are well-defined, speed matters, and accuracy is the primary metric. XGBoost trained in 2 minutes, infers in <1 second, and beat everything else on MAE.

**For this project:** Traditional ML (XGBoost) was the right choice for production predictions. Fine-tuning was educational but not competitive on accuracy.

### When does prompting alone get you close enough?

**Almost always for structured prediction tasks.** Our few-shot approach (3 examples in the prompt) achieved MAE 2.16 — within 2% of XGBoost. If we had tried this first, we might have concluded that fine-tuning wasn't worth the investment.

**Rule of thumb:** Before fine-tuning, always benchmark few-shot prompting. If it gets within 10% of your accuracy target, the engineering cost of fine-tuning probably isn't justified.

### What's the real cost of data preparation for fine-tuning?

The biggest hidden cost wasn't training time (20 minutes) — it was **discovering and fixing the mode collapse problem**. That took 3 iterations:
- v1: Naive training data → model collapsed to "always predict 2"
- v2: Bucket-balanced data → partial differentiation (4 values)
- v3: Per-point balanced data + doubled LoRA capacity → full range (6 values)

Each iteration required: re-engineering the data pipeline, retraining (~20 min), re-evaluating (~5 min), and diagnosing what went wrong (~30 min of analysis). The total cycle time was ~3 hours of human attention spread across 3 days.

### How do you evaluate a model for a noisy prediction task?

**Don't trust a single aggregate metric.** Our v1 fine-tuned model had a "decent" MAE because predicting "2" for every player is close to the mean. We only caught the problem by:
1. Inspecting the prediction *distribution* (all identical = mode collapse)
2. Looking at individual predictions (Salah = 2 pts? Obviously wrong)
3. Building category-based evals (captain picks, form streaks, etc.)

**The eval suite is the product spec.** Defining what "good" means for each scenario (captain picks, budget enablers, cold players) forced us to articulate what accuracy actually matters for user decisions.

### When does an LLM add value beyond raw accuracy?

Two cases where the LLM provided value despite worse MAE:

1. **Explanation capability.** Chain-of-thought predictions come with reasoning: "Home fixture against a leaky defence, strong form, but midweek Champions League rotation risk." XGBoost gives you a number; the LLM gives you a story.

2. **Range differentiation.** The fine-tuned v3 model predicts 9-10 for premium players in easy fixtures — something XGBoost's tighter range (typically 2-6) doesn't capture. For FPL decisions where you need to pick the *best* players (not just avoid the worst), wider differentiation has value.

### Based on your results, would you ship this? In what form?

**Yes — as a hybrid.** See the product brief below.

---

## 3. If We Had More Time

- **Larger model (7B or 14B):** More capacity might solve the fine-tuned accuracy gap. The 3B model may simply lack the parameters to learn both the range and the calibration.
- **Multi-season training data:** Currently training on ~30 gameweeks of one season. 3+ seasons would provide more edge cases and rare events.
- **Ensemble: XGBoost + LLM.** Use XGBoost's prediction as an additional feature in the LLM prompt — let the LLM adjust the statistical prediction with contextual reasoning.
- **Classification framing:** Instead of predicting exact points, predict tiers: "blank" (0-1), "return" (2-4), "haul" (5+). This matches how FPL managers actually think and might suit the LLM's discrete output nature.
- **Confidence calibration:** Our confidence scores are heuristic-based. With more data, we could calibrate them empirically: are "80% confidence" predictions actually right ~80% of the time?
