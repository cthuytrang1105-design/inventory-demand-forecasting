"""Random search trên validation theo thời gian với dự báo đệ quy từ một origin.

Không dùng CV trên bảng lag tính từ nhãn validation: cách đó dễ vô tình
biến đánh giá nhiều bước thành one-step với nhãn tương lai đã biết.
"""
import time
import pandas as pd
from sklearn.model_selection import ParameterSampler
from src.evaluate import join_actual, metrics
from src.feature_engineering import FEATURES
from src.models import build_model
from src.predict import recursive_forecast

SEARCH_SPACES = {
    "Ridge": {"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
    "RandomForest": {"model__max_depth": [6, 10, 16, None],
                     "model__min_samples_leaf": [1, 3, 6],
                     "model__max_features": [0.7, 1.0]},
    "HistGradientBoosting": {"model__learning_rate": [0.03, 0.06, 0.12],
                             "model__max_leaf_nodes": [7, 15, 31],
                             "model__l2_regularization": [0.0, 1.0, 10.0]},
}


def score_key(row):
    wape = row.get("WAPE_pct")
    return (row["MAE"] if wape is None or pd.isna(wape) else wape,
            row["inference_seconds"])


def search_model(name, train_features, history, validation, trials=3, quick=False):
    if trials < 1:
        raise ValueError("Số lượt tuning phải >= 1.")
    candidates = [({}, "before_tuning")]
    if name != "SeasonalNaive":
        candidates += [(params, "random_search") for params in ParameterSampler(
            SEARCH_SPACES[name], n_iter=trials, random_state=42)]
    records, best = [], None
    horizon = int((validation.date.max() - history.date.max()).days)
    for trial, (params, stage) in enumerate(candidates):
        model = build_model(name, params, quick)
        started = time.perf_counter()
        if model is not None:
            model.fit(train_features[FEATURES], train_features.sales)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        pred = recursive_forecast(model, history, horizon, name)
        inference_seconds = time.perf_counter() - started
        matched = join_actual(validation, pred)
        record = {"model": name, "trial": trial, "stage": stage, "params": params,
                  **metrics(matched.sales, matched.prediction),
                  "train_seconds": fit_seconds, "inference_seconds": inference_seconds,
                  "prediction_rows": len(pred)}
        records.append(record)
        if best is None or score_key(record) < score_key(best):
            best = record
    return best, records
