"""CLI: CSV lịch sử mới → kiểm tra → đặc trưng → model đã lưu → dự báo CSV."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.data_processing import KEYS, load_sales, regularize_daily, validate_sales
from src.feature_engineering import future_features
from src.utils import file_hash, read_metadata, write_json


def load_artifact(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Chưa có mô hình: {path}. Chạy python -m src.train theo README.")
    # joblib/pickle chỉ nạp artefact tin cậy do chính dự án huấn luyện.
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
        raise ValueError("Artefact không tương thích.")
    for key in ["model_name", "model", "series", "metadata"]:
        if key not in artifact:
            raise ValueError(f"Artefact thiếu {key}.")
    return artifact


def recursive_forecast(model, history, horizon=7, model_name=None):
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= 365:
        raise ValueError("Số ngày dự báo phải là số nguyên từ 1 đến 365.")
    history = validate_sales(history)
    end = history.date.max()
    daily = regularize_daily(history)
    keys, states = [], []
    for key, group in daily.groupby(KEYS, sort=True):
        if group.date.max() != end:
            raise ValueError(f"Chuỗi {key} chưa cập nhật tới {end.date()}; không tự dùng dữ liệu cũ.")
        if len(group) < 28:
            raise ValueError(f"Chuỗi {key} cần ít nhất 28 ngày lịch sử.")
        keys.append(key)
        states.append(group.sales.iloc[-28:].to_numpy(dtype=float))
    keys = pd.DataFrame(keys, columns=KEYS)
    state = np.vstack(states)
    outputs = []
    for date in pd.date_range(end + pd.Timedelta(days=1), periods=horizon):
        features = future_features(keys, date, state)
        prediction = state[:, -7].copy() if model_name == "SeasonalNaive" else model.predict(features)
        prediction = np.maximum(0, np.asarray(prediction, dtype=float))
        if not np.isfinite(prediction).all():
            raise ValueError("Mô hình trả dự báo không hữu hạn.")
        block = keys.copy()
        block["date"], block["prediction"] = date, prediction
        outputs.append(block)
        # Chỉ nối dự báo đã sinh; không bao giờ đọc nhãn của tương lai.
        state = np.column_stack([state[:, 1:], prediction])
    return pd.concat(outputs, ignore_index=True)[["date", *KEYS, "prediction"]]


def forecast(artifact, history, horizon=7):
    history = validate_sales(history)
    known = {tuple(x) for x in artifact["series"]}
    supplied = set(history[KEYS].itertuples(index=False, name=None))
    if not supplied.issubset(known):
        raise ValueError("Có cửa hàng/nhóm hàng chưa được huấn luyện: " + str(supplied - known))
    return recursive_forecast(artifact["model"], history, horizon, artifact["model_name"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="CSV lịch sử đã biết sales, KHÔNG phải test.csv của Kaggle")
    p.add_argument("--model", default="models/demo/model.joblib")
    p.add_argument("--horizon", type=int, choices=range(1, 366), default=7)
    p.add_argument("--store", type=int)
    p.add_argument("--family")
    p.add_argument("--output", default="results/forecast.csv")
    args = p.parse_args()
    artifact = load_artifact(args.model)
    df = load_sales(args.input)
    if args.store is not None:
        df = df[df.store_nbr == args.store]
    if args.family:
        df = df[df.family == args.family]
    result = forecast(artifact, df, args.horizon)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    write_json(path.with_suffix(".metadata.json"), {
        "history_source": read_metadata(args.input), "training_source": artifact["metadata"],
        "history_sha256": file_hash(args.input), "model_name": artifact["model_name"],
        "assumptions": "Không dùng khuyến mãi, dầu, giao dịch hoặc sự kiện tương lai. Recursive forecast; dự báo chặn tại 0.",
        "origin": str(df.date.max().date()), "horizon": args.horizon})
    print(f"Đã lưu {len(result)} dự báo tại {path}")


if __name__ == "__main__":
    main()
