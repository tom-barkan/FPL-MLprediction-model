# Product Brief: FPL Points Predictor

## Would I ship this? Yes — as a hybrid.

---

## 1. Recommended UX Based on Actual Model Accuracy

Our models have MAE 2.1–2.2 (XGBoost/few-shot), placing them in the **"show a range"** tier:

| Model Accuracy | Appropriate UX | Our Models |
|:---------------|:---------------|:-----------|
| MAE < 1.5 | Show exact prediction: "Salah: 8 pts" | Not there yet |
| **MAE 1.5–3.0** | **Show range: "Salah: 6–10 pts"** | **XGBoost (2.11), Few-shot (2.16)** |
| MAE > 3.0 | Show tier: "Salah: High return expected" | Fine-tuned v3 (3.35) |

**What we'd actually show in the app:**

```
Salah (MID) — Liverpool vs Bournemouth (H)
├── Predicted range: 6–10 pts (high confidence)
├── Model agreement: Both models agree — strong pick
└── Why: Home fixture, FDR 2, 8.3 avg form, 0.62 goals/90
```

The prediction range comes from XGBoost (central estimate) ± a margin based on confidence. The "Why" comes from the LLM's chain-of-thought reasoning. Neither model alone provides this — the hybrid does.

---

## 2. Cost of Being Wrong

In FPL, prediction errors have asymmetric consequences:

| Error Type | Scenario | Impact | Frequency |
|:-----------|:---------|:-------|:----------|
| **Low-stakes miss** | Predicted 4, actual 2 | User slightly disappointed | ~60% of errors |
| **Medium-stakes miss** | Captain pick blanks (predicted 8, got 2) | User loses 6-10 pts in their team | ~5% of errors |
| **Trust-breaking miss** | Model says "high confidence, captain this player" and they're benched | User stops using the tool | ~1% of errors |

**Accuracy threshold for shipping:** MAE < 3.0 with confidence calibration. Our XGBoost (2.11) clears this. The key protection is the confidence score — when confidence is low, the UX should downplay the prediction ("uncertain — consider alternatives").

**For higher-stakes domains (finance, health):** This accuracy level would not be shippable. The 17.7% of predictions outside ±3 points represents meaningful tail risk.

---

## 3. Build vs Buy vs Prompt

For this prediction task, we recommend:

```
[x] Hybrid: XGBoost for prediction + LLM for explanation and ranking
    → Because: XGBoost wins on numbers (MAE 2.11, <1s inference)
      but the LLM adds user-facing value through reasoning
      and player differentiation that XGBoost can't provide.

Supporting evidence:
- XGBoost: Best MAE (2.11), fastest (<1s), most consistent (82.3% within ±3)
- Few-shot LLM: Nearly as accurate (2.16) with zero training — good for explanation
- Fine-tuned LLM: Wider range (0-10) enables better ranking for squad selection
- Smoothed blend: 40% LLM + 30% form + 30% features gives 66 unique values

Rejected alternatives:
- Pure XGBoost: Accurate but can't explain "why" — users want reasoning
- Pure LLM: Too slow (370s+ per batch) and less accurate for production
- API-only (Claude/GPT): Few-shot works but adds per-prediction cost at scale
```

---

## 4. The Explanation Advantage

Chain-of-thought LLM output for a real prediction:

> **XGBoost says:** "Salah: 7.2 points"
> *No explanation. Take it or leave it.*

> **LLM says:** "Salah: 8 points — Home fixture against Bournemouth (FDR 2), the weakest away defence this season (1.8 goals conceded per GW). Salah's form has been excellent at 8.3 over the last 3 GWs with 0.62 goals per 90. Only risk: midweek Champions League fixture could mean rotation, though Salah has started every PL game this season."

The LLM is a slightly worse predictor but a significantly better product. Users don't just want a number — they want to understand *why*, so they can apply their own judgement. The explanation builds trust even when the prediction is wrong, because the user can see the reasoning and decide whether they agree.

**Assessment:** The explanation advantage justifies including the LLM in the product, even at the cost of complexity and slightly lower accuracy.

---

## 5. What It Would Take to Go to Production

### Data Freshness
- FPL API updates ~2 hours after each match finishes
- Pipeline: automated GW fetch → feature rebuild → XGBoost predict → LLM predict → dashboard update
- Need: cron job or webhook trigger, ~20 min end-to-end per gameweek

### Monitoring
- Track prediction accuracy per GW as the season progresses
- Alert if weekly MAE exceeds 3.5 (degradation signal)
- Monitor prediction distribution for mode collapse regression
- The eval suite provides the framework — run `compare.py` against baseline after each GW

### Edge Cases Not Yet Handled
- **New signings:** No historical features — fall back to position-average predictions
- **Double gameweeks:** Two fixtures in one GW — need to predict per-fixture and sum
- **Blank gameweeks:** Teams with no fixture — should predict 0 (not currently handled)
- **Injuries/suspensions:** Real-time availability data not in current pipeline
- **Manager changes:** Can shift team dynamics — not captured in rolling stats

### Scale
- Current: 344 players, single user, local inference
- At 10,000 users: XGBoost predictions are instant (serve from cache). LLM explanations could be pre-generated per GW and cached. Cost: ~$0 for XGBoost, ~$2-5/GW for LLM via API (if using Claude instead of local).
- Local MLX inference doesn't scale beyond single-machine — production would need API-based LLM or a GPU server

### Legal/Ethical
- FPL is a free-to-play game — low regulatory risk vs paid gambling
- Should include disclaimer: "Predictions are for entertainment. Past performance doesn't guarantee future results."
- No personal data collected beyond FPL public API data
- FPL API terms of service permit non-commercial use

---

## 6. Decision Matrix

| Consideration | XGBoost Only | LLM Only | Hybrid (Recommended) |
|:-------------|:------------|:---------|:---------------------|
| Accuracy | Best (2.11) | Good (2.16-2.20) | Best (uses XGBoost) |
| Speed | <1s | 370s+ | <1s predictions, LLM pre-cached |
| Explanations | None | Rich | Rich |
| Player ranking | Narrow range | Wide range | Blended (66 values) |
| Infra complexity | Low | High | Medium |
| Cost at scale | ~$0 | $2-5/GW | $2-5/GW |
| User trust | "Black box number" | "Here's my reasoning" | Both |

**The hybrid is the right answer.** Use XGBoost as the prediction engine, the LLM for explanations and ranking differentiation, and the smoothing pipeline to blend them. This gives users the accuracy they need and the context they want.
