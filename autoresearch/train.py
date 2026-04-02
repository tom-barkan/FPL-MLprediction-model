import xgboost as xgb
import numpy as np

PARAMS = {
    "n_estimators": 1500, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "objective": "reg:absoluteerror",
    "tree_method": "hist", "random_state": 42,
}

CLF_PARAMS = {
    "n_estimators": 500, "max_depth": 3, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
    "objective": "binary:logistic",
    "tree_method": "hist", "random_state": 42,
}

def engineer_features(X, feature_cols):
    return X.copy(), list(feature_cols)

def build_and_train(X_train, y_train, feature_cols):
    X_tr, used_cols = engineer_features(X_train, feature_cols)
    y_play = (y_train > 0).astype(int)
    clf = xgb.XGBClassifier(**CLF_PARAMS)
    clf.fit(X_tr[used_cols], y_play)
    reg = xgb.XGBRegressor(**PARAMS)
    reg.fit(X_tr[used_cols], y_train)
    return {"clf": clf, "reg": reg, "_used_cols": used_cols}

def predict(model, X_test, feature_cols):
    X_te, _ = engineer_features(X_test, feature_cols)
    used_cols = model["_used_cols"]
    play_prob = model["clf"].predict_proba(X_te[used_cols])[:, 1]
    reg_pred = model["reg"].predict(X_te[used_cols])
    # Softer blend: 0.8 * reg + 0.2 * (prob * reg)
    return (0.5 + 0.5 * play_prob) * reg_pred
