# Eval Run: v3-baseline
**Date:** 2026-03-26
**Test set:** 243 player-gameweeks

## Overall

| Model | MAE | RMSE | Within ±1 | Within ±3 |
|:------|----:|-----:|----------:|----------:|
| xgboost | 2.106 | 2.806 | 26.7% | 82.3% |
| llm_zero_shot | 11.539 | 14.221 | 3.3% | 21.0% |
| llm_few_shot | 2.165 | 3.320 | 55.6% | 76.1% |
| llm_chain_of_thought | 2.198 | 3.597 | 60.1% | 73.7% |
| llm_fine_tuned | 3.350 | 4.458 | 39.1% | 65.8% |

## Category Breakdown

### edge_cases

| Case | xgboost | llm_zero_shot | llm_few_shot | llm_chain_of_thought | llm_fine_tuned | Count |
|:-----|------:|------:|------:|------:|------:|------:|
| cold_player | 1.645 | 10.224 | 1.424 | 1.341 | 2.224 | 85 |
| player_low_minutes | 1.945 | 10.609 | 1.773 | 1.625 | 2.945 | 128 |
| player_on_hot_streak | 3.287 | 19.000 | 3.750 | 4.000 | 6.250 | 4 |
| struggling_team | 1.768 | 9.806 | 1.677 | 1.548 | 2.855 | 62 |

### fixture_context

| Case | xgboost | llm_zero_shot | llm_few_shot | llm_chain_of_thought | llm_fine_tuned | Count |
|:-----|------:|------:|------:|------:|------:|------:|
| home_vs_away_all | 2.192 | 12.408 | 2.224 | 2.384 | 3.472 | 125 |

### high_value

| Case | xgboost | llm_zero_shot | llm_few_shot | llm_chain_of_thought | llm_fine_tuned | Count |
|:-----|------:|------:|------:|------:|------:|------:|
| budget_enablers | 2.287 | 10.690 | 2.333 | 2.738 | 4.024 | 42 |
| captain_candidates | 4.281 | 13.333 | 5.000 | 4.333 | 7.000 | 3 |

### positional

| Case | xgboost | llm_zero_shot | llm_few_shot | llm_chain_of_thought | llm_fine_tuned | Count |
|:-----|------:|------:|------:|------:|------:|------:|
| budget_defender_away | 2.247 | 5.704 | 1.926 | 2.111 | 2.148 | 27 |
| goalkeeper_home | 1.579 | 7.250 | 2.250 | 2.875 | 2.375 | 8 |
| premium_forward_form | 2.912 | 23.000 | 4.000 | 0.000 | 9.000 | 1 |

## Output Quality (LLM models)

- **llm_zero_shot**: 5 unique values ([4, 5, 8, 11, 24]) — PASS
- **llm_few_shot**: 7 unique values ([0, 1, 2, 3, 4, 5, 9]) — PASS
- **llm_chain_of_thought**: 2 unique values ([0, 1]) — PASS
- **llm_fine_tuned**: 6 unique values ([0, 1, 2, 4, 9, 10]) — PASS
