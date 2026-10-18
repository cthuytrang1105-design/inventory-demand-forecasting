"""Đặc trưng dùng cùng công thức ở train và recursive inference."""
import numpy as np
import pandas as pd
from src.data_processing import KEYS, regularize_daily

LAGS = (1, 7, 14, 28)
WINDOWS = (7, 14, 28)
CALENDAR = ["day", "month", "weekday", "weekend", "dow_sin", "dow_cos",
            "year_sin", "year_cos", "trend"]
HISTORY = [f"lag_{lag}" for lag in LAGS] + [f"rolling_{stat}_{w}" for w in WINDOWS for stat in ("mean", "std")]
NUMERIC = CALENDAR + HISTORY
FEATURES = KEYS + NUMERIC


def add_calendar(frame):
    df = frame.copy()
    d = pd.to_datetime(df["date"])
    df["day"], df["month"], df["weekday"] = d.dt.day, d.dt.month, d.dt.dayofweek
    df["weekend"] = (d.dt.dayofweek >= 5).astype(int)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * d.dt.dayofweek / 7), np.cos(2 * np.pi * d.dt.dayofweek / 7)
    df["year_sin"], df["year_cos"] = np.sin(2 * np.pi * d.dt.dayofyear / 365.25), np.cos(2 * np.pi * d.dt.dayofyear / 365.25)
    df["trend"] = (d - pd.Timestamp("2020-01-01")).dt.days
    return df


def make_features(raw):
    df = add_calendar(regularize_daily(raw))
    groups = df.groupby(KEYS, sort=False)["sales"]
    for lag in LAGS:
        df[f"lag_{lag}"] = groups.shift(lag)
    for w in WINDOWS:
        # Dịch trước rolling: tuyệt đối không chứa sales của chính ngày t.
        df[f"rolling_mean_{w}"] = groups.transform(lambda s: s.shift(1).rolling(w, min_periods=w).mean())
        df[f"rolling_std_{w}"] = groups.transform(lambda s: s.shift(1).rolling(w, min_periods=w).std(ddof=0))
    return df


def future_features(keys, date, history):
    """history: ma trận n_series × 28 ngày gần nhất, thứ tự cũ → mới."""
    if history.ndim != 2 or history.shape[1] != 28 or not np.isfinite(history).all():
        raise ValueError("Cần 28 ngày lịch sử hữu hạn cho mỗi chuỗi.")
    df = keys.copy().reset_index(drop=True)
    df["date"] = pd.Timestamp(date)
    df = add_calendar(df)
    for lag in LAGS:
        df[f"lag_{lag}"] = history[:, -lag]
    for w in WINDOWS:
        df[f"rolling_mean_{w}"] = history[:, -w:].mean(axis=1)
        df[f"rolling_std_{w}"] = history[:, -w:].std(axis=1, ddof=0)
    return df[FEATURES]
