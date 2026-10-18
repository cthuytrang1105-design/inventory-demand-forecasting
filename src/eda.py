"""Xuất báo cáo chất lượng và biểu đồ HTML từ CSV thực có."""
import argparse
from pathlib import Path
import pandas as pd
import plotly.express as px
from src.data_processing import KEYS, load_sales, quality_summary
from src.utils import read_metadata, write_json


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
              "interpretation": "Mô tả tương quan, không suy luận tác động nhân quả của khuyến mãi/ngày lễ."}
    df["weekday"] = df.date.dt.dayofweek
    df["month"] = df.date.dt.month
    figures = {
        "eda_daily": px.line(df.groupby("date", as_index=False).sales.sum(), x="date", y="sales", title="Xu hướng tổng sales theo ngày"),
        "eda_store": px.bar(df.groupby("store_nbr", as_index=False).sales.sum(), x="store_nbr", y="sales", title="Tổng sales theo cửa hàng"),
        "eda_family": px.bar(df.groupby("family", as_index=False).sales.sum(), x="family", y="sales", title="Tổng sales theo nhóm hàng"),
        "eda_weekday": px.bar(df.groupby("weekday", as_index=False).sales.mean(), x="weekday", y="sales", title="Sales trung bình theo thứ (0 = thứ Hai)"),
        "eda_month": px.bar(df.groupby("month", as_index=False).sales.mean(), x="month", y="sales", title="Sales trung bình theo tháng"),
        "eda_promotion": px.box(df.assign(promotion=df.onpromotion.gt(0).map({True: "Có", False: "Không"})), x="promotion", y="sales", title="Phân phối sales theo khuyến mãi (tương quan)"),
    }
    if holidays:
        enriched, note = holiday_effect(df, holidays, stores)
        figures["eda_holiday"] = px.box(enriched, x="holiday", y="sales", title="Sales và ngày lễ theo địa bàn")
        report["holiday_analysis"] = note
    else:
        report["holiday_analysis"] = "Chưa có holidays_events.csv; không tự gán ngày lễ từ dữ liệu không tồn tại."
    for name, fig in figures.items():
        fig.update_layout(xaxis_title={"date": "Ngày", "store_nbr": "Cửa hàng", "family": "Nhóm hàng", "weekday": "Thứ trong tuần", "month": "Tháng", "promotion": "Có khuyến mãi", "holiday": "Có ngày lễ"}.get(fig.layout.xaxis.title.text, fig.layout.xaxis.title.text), yaxis_title="sales (đơn vị gốc)")
        fig.write_html(str(output / "figures" / f"{name}.html"), include_plotlyjs="directory")
    write_json(output / "metrics/eda.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="results/demo")
    p.add_argument("--holidays")
    p.add_argument("--stores")
    a = p.parse_args()
    print(run_eda(a.data, a.output, a.holidays, a.stores))
