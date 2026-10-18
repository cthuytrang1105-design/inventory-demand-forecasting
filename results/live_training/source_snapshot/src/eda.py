"""Xuất báo cáo chất lượng và biểu đồ HTML từ CSV thực có."""
import argparse
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data_processing import KEYS, load_sales, quality_summary
from src.utils import read_metadata, write_json


def distribution_summary(df, column):
    """Tính quantile trên TOÀN BỘ dữ liệu; chỉ xuất thống kê, không gửi hàng triệu điểm ra HTML."""
    rows = []
    for label, group in df.groupby(column, sort=True):
        sales = group.sales
        q1, median, q3 = sales.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        inside = sales[sales.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
        rows.append({column: label, "count": len(sales), "mean": float(sales.mean()),
                     "q1": float(q1), "median": float(median), "q3": float(q3),
                     "lowerfence": float(inside.min()), "upperfence": float(inside.max())})
    return pd.DataFrame(rows)


def summary_box(summary, column, title):
    fig = go.Figure(go.Box(x=summary[column].astype(str), q1=summary.q1, median=summary["median"],
                           q3=summary.q3, lowerfence=summary.lowerfence, upperfence=summary.upperfence,
                           boxpoints=False, name="Toàn bộ dữ liệu"))
    fig.update_layout(title=title, xaxis_title=column)
    return fig


def holiday_effect(df, holidays_path, stores_path=None):
    events = pd.read_csv(holidays_path)
    required = {"date", "type", "locale", "locale_name", "transferred"}
    if not required.issubset(events.columns):
        raise ValueError("holidays_events.csv thiếu cột: " + str(required - set(events.columns)))
    events["date"] = pd.to_datetime(events.date)
    transferred = events.transferred.astype(str).str.lower().isin(["true", "1"])
    events = events[~transferred & events.type.ne("Work Day")]
    enriched = df.copy()
    if stores_path:
        stores = pd.read_csv(stores_path)
        if not {"store_nbr", "city", "state"}.issubset(stores.columns):
            raise ValueError("stores.csv cần store_nbr, city, state.")
        enriched = enriched.merge(stores[["store_nbr", "city", "state"]], on="store_nbr", how="left", validate="many_to_one")
        if enriched[["city", "state"]].isna().any().any():
            raise ValueError("stores.csv thiếu địa bàn cho một số cửa hàng.")
    matches = enriched[["date", *KEYS] + (["city", "state"] if stores_path else [])].merge(events, on="date", how="inner")
    mask = matches.locale.eq("National")
    if stores_path:
        mask |= matches.locale.eq("Local") & matches.locale_name.eq(matches.city)
        mask |= matches.locale.eq("Regional") & matches.locale_name.eq(matches.state)
    flagged = matches.loc[mask, ["date", *KEYS]].drop_duplicates().assign(holiday=True)
    enriched = enriched.merge(flagged, on=["date", *KEYS], how="left")
    enriched["holiday"] = enriched.holiday.eq(True)
    return enriched, "Ghép đúng địa bàn; loại ngày transferred và Work Day." if stores_path else "Chỉ ngày lễ National; cung cấp stores.csv để ghép Regional/Local."


def run_eda(path, output="results/demo", holidays=None, stores=None):
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(path)
    df = load_sales(path)
    report = {"source": read_metadata(path), **quality_summary(raw),
              "date_min": str(df.date.min().date()), "date_max": str(df.date.max().date()),
              "stores": int(df.store_nbr.nunique()), "families": int(df.family.nunique()),
              "sales_summary": {k: float(v) for k, v in df.sales.describe().items()},
              "zero_sales_rows": int(df.sales.eq(0).sum()),
              "zero_sales_pct": float(df.sales.eq(0).mean() * 100),
              "missing_calendar_dates": pd.date_range(df.date.min(), df.date.max()).difference(df.date.unique()).strftime("%Y-%m-%d").tolist(),
              "interpretation": "Mô tả tương quan, không suy luận tác động nhân quả của khuyến mãi/ngày lễ."}
    df["weekday"] = df.date.dt.dayofweek
    df["month"] = df.date.dt.month
    summaries = {"daily": df.groupby("date", as_index=False).sales.sum(),
                 "store": df.groupby("store_nbr", as_index=False).sales.sum(),
                 "family": df.groupby("family", as_index=False).sales.sum(),
                 "weekday": df.groupby("weekday", as_index=False).sales.mean(),
                 "month": df.groupby("month", as_index=False).sales.mean()}
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    for name, frame in summaries.items():
        frame.to_csv(output / "metrics" / f"eda_{name}.csv", index=False)
    promo_summary = distribution_summary(df.assign(promotion=df.onpromotion.gt(0).map({True: "Có", False: "Không"})), "promotion")
    report["promotion_summary"] = promo_summary.to_dict(orient="records")
    figures = {
        "eda_daily": px.line(df.groupby("date", as_index=False).sales.sum(), x="date", y="sales", title="Xu hướng tổng sales theo ngày"),
        "eda_store": px.bar(df.groupby("store_nbr", as_index=False).sales.sum(), x="store_nbr", y="sales", title="Tổng sales theo cửa hàng"),
        "eda_family": px.bar(df.groupby("family", as_index=False).sales.sum(), x="family", y="sales", title="Tổng sales theo nhóm hàng"),
        "eda_weekday": px.bar(df.groupby("weekday", as_index=False).sales.mean(), x="weekday", y="sales", title="Sales trung bình theo thứ (0 = thứ Hai)"),
        "eda_month": px.bar(df.groupby("month", as_index=False).sales.mean(), x="month", y="sales", title="Sales trung bình theo tháng"),
        "eda_promotion": summary_box(promo_summary, "promotion", "Sales theo khuyến mãi · quantile toàn bộ dữ liệu"),
    }
    if holidays:
        enriched, note = holiday_effect(df, holidays, stores)
        holiday_summary = distribution_summary(enriched, "holiday")
        figures["eda_holiday"] = summary_box(holiday_summary, "holiday", "Sales và ngày lễ theo địa bàn · quantile toàn bộ dữ liệu")
        report["holiday_analysis"] = note
        report["holiday_summary"] = holiday_summary.to_dict(orient="records")
    else:
        report["holiday_analysis"] = "Chưa có holidays_events.csv; không tự gán ngày lễ từ dữ liệu không tồn tại."
    for name, fig in figures.items():
        fig.update_layout(xaxis_title={"date": "Ngày", "store_nbr": "Cửa hàng", "family": "Nhóm hàng", "weekday": "Thứ trong tuần", "month": "Tháng", "promotion": "Có khuyến mãi", "holiday": "Có ngày lễ"}.get(fig.layout.xaxis.title.text, fig.layout.xaxis.title.text), yaxis_title="sales (đơn vị gốc)")
        fig.write_html(str(output / "figures" / f"{name}.html"), include_plotlyjs="directory")
    write_json(output / "metrics/eda.json", report)
    write_json(output / "metrics/eda_distributions.json", {
        "promotion": promo_summary.to_dict(orient="records"),
        "holiday": report.get("holiday_summary", []),
        "method": "Quartile và whisker 1.5 IQR tính trên toàn bộ dữ liệu; không lấy mẫu, không vẽ từng outlier."})
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="results/demo")
    p.add_argument("--holidays")
    p.add_argument("--stores")
    a = p.parse_args()
    print(run_eda(a.data, a.output, a.holidays, a.stores))
