"""Sinh dữ liệu MÔ PHỎNG, tuyệt đối không trình bày như dữ liệu Kaggle."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import write_json


def generate_demo(path="data/demo/sales.csv", days=480, seed=42):
    if days < 120:
        raise ValueError("Demo cần tối thiểu 120 ngày.")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days)
    rows = []
    for store in (1, 2, 3):
        for family, base in (("GROCERY I", 120), ("BEVERAGES", 85), ("PRODUCE", 65)):
            for t, day in enumerate(dates):
                promotion = int(rng.random() < 0.12) * int(rng.integers(1, 8))
                holiday = day.month == 12 and day.day in (24, 25, 31)
                sales = max(0, base * (1 + 0.12 * store) + 0.035 * t
                            + 18 * np.sin(2 * np.pi * day.dayofweek / 7)
                            + 9 * np.sin(2 * np.pi * day.dayofyear / 365.25)
                            + 1.7 * promotion + 20 * holiday + rng.normal(0, 5))
                rows.append((day, store, family, round(sales, 3), promotion))
    df = pd.DataFrame(rows, columns=["date", "store_nbr", "family", "sales", "onpromotion"])
    df.insert(0, "id", np.arange(len(df)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    write_json(path.with_suffix(".metadata.json"), {
        "source": "DỮ LIỆU MÔ PHỎNG — tạo bởi src.demo_data, không phải Kaggle",
        "simulated": True, "seed": seed, "days": days,
        "unit": "đơn vị nhu cầu mô phỏng; không đại diện SKU thực"})
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tạo dữ liệu mô phỏng có nhãn nguồn")
    p.add_argument("--output", default="data/demo/sales.csv")
    p.add_argument("--days", type=int, default=480)
    args = p.parse_args()
    print(f"Đã tạo {len(generate_demo(args.output, args.days)):,} dòng MÔ PHỎNG tại {args.output}")
