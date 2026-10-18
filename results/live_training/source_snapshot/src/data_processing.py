"""Kiểm tra schema thực tế; không âm thầm biến nhãn thiếu thành doanh số 0."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["store_nbr", "family"]
REQUIRED = ["date", *KEYS, "sales"]


def validate_sales(frame):
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError("CSV thiếu cột: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("Dữ liệu rỗng.")
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    if df["date"].isna().any() or df["date"].dt.tz is not None:
        raise ValueError("Ngày không hợp lệ hoặc có múi giờ; dùng YYYY-MM-DD.")
    if not df["date"].eq(df["date"].dt.normalize()).all():
        raise ValueError("Dữ liệu phải có độ phân giải ngày, không chứa giờ.")
    store = pd.to_numeric(df["store_nbr"], errors="coerce")
    if not np.isfinite(store).all() or (store < 1).any() or (store % 1 != 0).any():
        raise ValueError("store_nbr phải là số nguyên dương.")
    df["store_nbr"] = store.astype(int)
    if df["family"].isna().any() or df["family"].astype(str).str.strip().eq("").any():
        raise ValueError("family không được thiếu hoặc rỗng.")
    df["family"] = df["family"].astype(str).str.strip()
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    if not np.isfinite(df["sales"]).all() or (df["sales"] < 0).any():
        raise ValueError("sales phải hữu hạn, không âm và không thiếu; không tự tạo nhãn.")
    if df.duplicated(["date", *KEYS]).any():
        raise ValueError("Trùng khóa ngày × cửa hàng × nhóm hàng; cần xử lý tại nguồn.")
    if "onpromotion" not in df:
        df["onpromotion"] = 0.0
    df["onpromotion"] = pd.to_numeric(df["onpromotion"], errors="coerce").fillna(0)
    if not np.isfinite(df["onpromotion"]).all() or (df["onpromotion"] < 0).any():
        raise ValueError("onpromotion phải hữu hạn và không âm.")
    return df.sort_values([*KEYS, "date"]).reset_index(drop=True)


def load_sales(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy {path}. Xem README để tạo demo hoặc tải Kaggle.")
    return validate_sales(pd.read_csv(path))


def regularize_daily(df):
    """Chèn ngày thiếu để lag_7 đúng 7 ngày; ffill chỉ làm trạng thái quá khứ.

    Ngày chèn KHÔNG được dùng làm nhãn huấn luyện hoặc ground truth đánh giá.
    Không kéo dài series quá ngày cuối thực có của series đó.
    """
    blocks = []
    for key, group in df.groupby(KEYS, sort=True):
        g = group.set_index("date").sort_index()
        grid = pd.date_range(g.index.min(), g.index.max(), freq="D", name="date")
        observed = pd.Series(True, index=g.index).reindex(grid, fill_value=False)
        g = g.reindex(grid)
        g["is_observed"] = observed
        g["sales"] = g["sales"].ffill()
        g["onpromotion"] = g["onpromotion"].fillna(0)
        for col, value in zip(KEYS, key):
            g[col] = value
        blocks.append(g.reset_index())
    return pd.concat(blocks, ignore_index=True)


def quality_summary(frame):
    return {"rows": len(frame), "columns": list(frame.columns),
            "missing_values": {str(k): int(v) for k, v in frame.isna().sum().items()},
            "duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_keys": int(frame.duplicated(["date", *KEYS]).sum())
            if set(["date", *KEYS]).issubset(frame.columns) else None}


def split_dates(df, horizon=30):
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon phải là số nguyên dương.")
    dates = pd.date_range(df.date.min(), df.date.max(), freq="D")
    if len(dates) < 2 * horizon + 60:
        raise ValueError("Cần ít nhất 60 + 2 × horizon ngày để chia train/validation/test.")
    val_start, test_start = dates[-2 * horizon], dates[-horizon]
    return (df[df.date < val_start].copy(),
            df[(df.date >= val_start) & (df.date < test_start)].copy(),
            df[df.date >= test_start].copy())


def main():
    p = argparse.ArgumentParser(description="Kiểm tra CSV bán hàng")
    p.add_argument("--data", required=True)
    args = p.parse_args()
    raw = pd.read_csv(args.data)
    print(quality_summary(raw))
    df = validate_sales(raw)
    print(f"Hợp lệ: {len(df):,} dòng; {df.date.min().date()} → {df.date.max().date()}")


if __name__ == "__main__":
    main()
