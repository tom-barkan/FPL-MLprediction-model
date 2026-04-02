import sys, json, time, importlib.util, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_data, evaluate, load_baseline, save_baseline

train_path = Path(__file__).parent / "train.py"
spec = importlib.util.spec_from_file_location("train_mod", train_path)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

X_train, y_train, X_test, y_test, feature_cols = load_data()

print("Training...")
t0 = time.time()
model = train_mod.build_and_train(X_train, y_train, feature_cols)
elapsed = time.time() - t0

print("Evaluating...")
y_pred = train_mod.predict(model, X_test, feature_cols)
metrics = evaluate(y_test.values, y_pred)
metrics["train_time_s"] = round(elapsed, 1)

baseline_mae = load_baseline()
new_mae = metrics["mae"]

print(f"\n── Result ───────────────────────────")
print(f"MAE:        {new_mae}  (baseline: {baseline_mae})")
print(f"RMSE:       {metrics['rmse']}")
print(f"Within ±1:  {metrics['within_1_pct']}%")
print(f"Within ±3:  {metrics['within_3_pct']}%")
print(f"Train time: {metrics['train_time_s']}s")

if new_mae < baseline_mae:
    save_baseline(metrics)
    print(f"\n✅ IMPROVEMENT: {baseline_mae} → {new_mae}")
    sys.exit(0)
else:
    print(f"\n❌ NO IMPROVEMENT: {new_mae} vs {baseline_mae}")
    sys.exit(1)
