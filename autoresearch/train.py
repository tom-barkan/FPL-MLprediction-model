import xgboost as xgb
import numpy as np

PARAMS = {
    "n_estimators": 700, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "objective": "reg:absoluteerror",
    "tree_method": "hist", "random_state": 42,
}

def engineer_features(X, feature_cols):
    return X.copy(), list(feature_cols)

def build_and_train(X_train, y_train, feature_cols):
    X_tr, used_cols = engineer_features(X_train, feature_cols)
    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X_tr[used_cols], y_train)
    model._used_cols = used_cols
    return model

def predict(model, X_test, feature_cols):
    X_te, _ = engineer_features(X_test, feature_cols)
    used_cols = getattr(model, "_used_cols", list(feature_cols))
    return model.predict(X_te[used_cols])
