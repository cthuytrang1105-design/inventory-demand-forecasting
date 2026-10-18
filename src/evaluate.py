"""Metric trên holdout và CLI tái tính từ dự báo đã lưu, không train lại."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.data_processing import KEYS
from src.utils import write_json


def metrics(actual, predicted):
    y, pred = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y.shape != pred.shape or y.size == 0 or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Nhãn và dự báo phải cùng kích thước, khác rỗng và hữu hạn.")
    errors = np.abs(y - pred)
    denominator = float(np.abs(y).sum())
    return {"MAE": float(errors.mean()), "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
            "WAPE_pct": float(100 * errors.sum() / denominator) if denominator else None}


def join_actual(actual, prediction):
    result = actual[["date", *KEYS, "sales"]].merge(prediction, on=["date", *KEYS], how="left", validate="one_to_one")
    if result.empty or result.prediction.isna().any():
        raise ValueError("Không đủ dự báo cho mọi quan sát thật trong holdout.")
    return result


def export_test_figure(predictions, path):
    import plotly.express as px
    aggregate = predictions.groupby("date", as_index=False)[["sales", "prediction"]].sum()
    chart = px.line(aggregate, x="date", y=["sales", "prediction"],
                    title="Tổng nhu cầu test: thực tế và dự báo đệ quy",
                    labels={"value": "sales (đơn vị gốc)", "date": "Ngày", "variable": "Chuỗi"})
    chart.for_each_trace(lambda t: t.update(name={"sales": "Thực tế", "prediction": "Dự báo"}[t.name]))
    chart.write_html(str(path), include_plotlyjs=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default="results/demo/metrics/test_predictions.csv")
    p.add_argument("--output", default="results/demo/metrics/recomputed_test.json")
    args = p.parse_args()
    df = pd.read_csv(args.predictions)
    out = metrics(df.sales, df.prediction)
    write_json(args.output, out)
    print(out)


if __name__ == "__main__":
    main()
