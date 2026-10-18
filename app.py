"""Streamlit tiếng Việt; chỉ đọc artefact, tuyệt đối không tự train trong giao diện."""
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from threadpoolctl import threadpool_limits
from src.data_processing import KEYS, load_sales
from src.inventory import recommend_order
from src.predict import forecast, load_artifact
from src.utils import read_metadata

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Dự báo nhu cầu | DemandLab", page_icon="📦", layout="wide")
st.markdown("""<style>
    .stApp h1, .stApp h2, .stApp h3 {color:#142b50;}
    [data-testid="stMetric"] {background:white;border:1px solid #e5eaf3;border-radius:12px;padding:16px;}
    [data-testid="stSidebar"] {border-right:1px solid #e5eaf3;}
    .block-container {padding-top:2rem;}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def cached_data(path, modified):
    return load_sales(path)


@st.cache_resource(show_spinner=False)
def cached_model(path, modified):
    return load_artifact(path)


def show_table(frame):
    labels = {"date": "Ngày", "store_nbr": "Cửa hàng", "family": "Nhóm hàng", "sales": "Thực tế",
              "prediction": "Dự báo", "model": "Thuật toán", "selected": "Được chọn",
              "train_seconds": "Huấn luyện (giây)", "inference_seconds": "Suy luận (giây)",
              "inference_ms_per_row": "Suy luận (ms/dòng)", "params": "Siêu tham số"}
    labels.update({f"{part}_{metric}": f"{metric} · {vietnamese}"
                   for part, vietnamese in [("validation", "kiểm định"), ("test", "kiểm tra")]
                   for metric in ["MAE", "RMSE", "WAPE_pct"]})
    st.dataframe(frame.rename(columns=labels), hide_index=True, width="stretch")


def format_chart(fig, ylabel="sales (đơn vị gốc)"):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=45, b=20),
                      xaxis_title="Ngày", yaxis_title=ylabel,
                      colorway=["#2563eb", "#0d9488", "#f59e0b"])
    st.plotly_chart(fig, use_container_width=True)


def selection(df, prefix):
    col1, col2 = st.columns(2)
    store = col1.selectbox("Cửa hàng", sorted(df.store_nbr.unique()), key=f"{prefix}_store")
    families = sorted(df.loc[df.store_nbr == store, "family"].unique())
    family = col2.selectbox("Nhóm sản phẩm", families, key=f"{prefix}_family")
    return df[(df.store_nbr == store) & (df.family == family)].copy()


with st.sidebar:
    st.title("📦 DemandLab")
    st.caption("DỰ BÁO NHU CẦU & HỖ TRỢ TỒN KHO")
    mode = st.selectbox("Nguồn chạy", ["Demo mô phỏng", "CSV người dùng / Kaggle"],
                        index=1 if (ROOT / "results/official/run_manifest.json").exists() else 0)
    run_name = "demo" if mode == "Demo mô phỏng" else "official"
    page = st.radio("Điều hướng", ["Tổng quan", "Phân tích dữ liệu", "Dự báo nhu cầu",
                                     "Quản lý tồn kho", "Đánh giá mô hình", "Giới thiệu hệ thống"])
    st.divider()
    st.caption("Huấn luyện độc lập • Suy luận cục bộ\n\nCửa hàng × nhóm hàng × ngày")

result_dir = ROOT / "results" / run_name
manifest_path = result_dir / "run_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
default_path = "data/demo/sales.csv" if run_name == "demo" else "data/raw/train.csv"
if manifest:
    default_path = manifest["metadata"]["data_path"]
data_path = st.sidebar.text_input("Đường dẫn CSV lịch sử", default_path)
st.title(page)
st.caption("Hệ thống dự báo nhu cầu hàng hóa hỗ trợ lập kế hoạch nhập hàng bằng Machine Learning")

data = None
artifact = None
try:
    absolute = ROOT / data_path
    data = cached_data(str(absolute), absolute.stat().st_mtime_ns)
    source = read_metadata(absolute)
    if source.get("simulated"):
        st.warning("DỮ LIỆU MÔ PHỎNG — dùng để kiểm thử và minh họa, không phải kết quả trên Kaggle.")
    else:
        st.info(f"{source['source']}. Dữ liệu bán hàng không chứa tồn kho thực tế.")
    if manifest:
        model_path = ROOT / manifest["model_path"]
        artifact = cached_model(str(model_path), model_path.stat().st_mtime_ns)
        if artifact["metadata"].get("simulated"):
            st.caption("Mô hình hiện tại được huấn luyện trên dữ liệu mô phỏng.")
except (ValueError, FileNotFoundError, OSError) as exc:
    st.error(str(exc))


if page == "Giới thiệu hệ thống":
    st.markdown("""### Từ lịch sử bán hàng đến kế hoạch bổ sung
    Dự án học các quy luật nhu cầu theo lịch và lịch sử của từng cửa hàng, nhóm hàng.
    Mục tiêu là so sánh các phương pháp ML và biến dự báo thành thông tin hỗ trợ quản lý.

    **Quy trình:** CSV → kiểm tra → đặc trưng quá khứ → train / validation / test →
    tuning → chọn mô hình → dự báo đệ quy → quy tắc nhập hàng.

    **Công nghệ:** Python, pandas, scikit-learn, Plotly và Streamlit.
    Ba thuật toán: Ridge, Random Forest, HistGradientBoosting; thêm seasonal naive.
    Không sử dụng API mô hình ngôn ngữ lớn.

    **Phạm vi:** dự báo `sales` ở cấp cửa hàng × nhóm hàng × ngày, không phải từng SKU.
    Dữ liệu này không cung cấp tồn kho thực; mọi thông tin tồn kho do người dùng nhập.
    `sales` tổng hợp nhóm hàng có thể bao gồm nhiều đơn vị khác nhau; cần quy đổi hợp lệ
    trước khi dùng cho kế hoạch tồn kho thực tế.

    **Giới hạn:** chưa tính chi phí tối ưu, chưa có khoảng dự báo, chưa mô hình hóa tác động
    khuyến mãi/ngày lễ trong mô hình. Các biến ngoại sinh tương lai được loại khỏi đặc trưng.
    Chỉ có một origin validation/test. Đây là hỗ trợ quyết định, không tự động đặt hàng.
    """)
    st.link_button("Dataset Store Sales trên Kaggle", "https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data")
elif data is None:
    st.info("Tạo dữ liệu demo bằng `python -m src.demo_data` hoặc đặt train.csv vào data/raw/.")
elif page == "Tổng quan":
    cols = st.columns(4)
    for col, name, value in zip(cols, ["Quan sát lịch sử", "Cửa hàng", "Nhóm sản phẩm", "Tổng sales"],
                                 [f"{len(data):,}", data.store_nbr.nunique(), data.family.nunique(), f"{data.sales.sum():,.1f}"]):
        col.metric(name, value)
    st.caption(f"{data.date.min():%d/%m/%Y} — {data.date.max():%d/%m/%Y} · sales theo đơn vị gốc · không phải doanh thu tiền tệ")
    daily = data.groupby("date", as_index=False).sales.sum()
    format_chart(px.line(daily, x="date", y="sales", title="Xu hướng tổng nhu cầu bán hàng"))
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Mô hình đang sử dụng")
        if manifest and artifact:
            st.write(f"**{manifest['selected_model']}** · {manifest['metadata']['series_count']} chuỗi được huấn luyện")
            st.write(manifest["selection_reason"])
            st.caption(f"Dữ liệu fit gần nhất: {artifact['training_end']}")
            families = sorted({pair[1] for pair in manifest["series"]})
            st.caption(f"Phạm vi mô hình: {len({pair[0] for pair in manifest['series']})} cửa hàng · "
                       + ", ".join(families))
        else:
            st.info("Chưa có mô hình. Chạy lệnh huấn luyện trong README.")
    with right:
        st.subheader("Nguồn và cách đọc dữ liệu")
        st.write(source["source"])
        st.caption("Phần phân tích có thể bao gồm nhiều chuỗi hơn mô hình quick. Trang dự báo chỉ cho chọn chuỗi đã học.")
elif page == "Phân tích dữ liệu":
    subset = selection(data, "eda")
    interval = st.date_input("Khoảng ngày", (subset.date.min().date(), subset.date.max().date()),
                             min_value=subset.date.min().date(), max_value=subset.date.max().date())
    if len(interval) == 2:
        subset = subset[subset.date.between(pd.Timestamp(interval[0]), pd.Timestamp(interval[1]))]
    if subset.empty:
        st.info("Không có quan sát trong khoảng đã chọn.")
    else:
        cols = st.columns(3)
        cols[0].metric("Số quan sát đang xem", len(subset))
        cols[1].metric("sales trung bình / ngày", f"{subset.sales.mean():,.2f}")
        cols[2].metric("sales lớn nhất / ngày", f"{subset.sales.max():,.2f}")
        format_chart(px.line(subset, x="date", y="sales", title="Xu hướng theo bộ lọc"))
        left, right = st.columns(2)
        with left:
            by_day = subset.assign(weekday=subset.date.dt.dayofweek).groupby("weekday", as_index=False).sales.mean()
            fig = px.bar(by_day, x="weekday", y="sales", title="Nhu cầu theo thứ · 0 = thứ Hai")
            fig.update_layout(xaxis_title="Thứ trong tuần", yaxis_title="sales trung bình")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            by_month = subset.assign(month=subset.date.dt.month).groupby("month", as_index=False).sales.mean()
            fig = px.bar(by_month, x="month", y="sales", title="Nhu cầu theo tháng")
            fig.update_layout(xaxis_title="Tháng", yaxis_title="sales trung bình")
            st.plotly_chart(fig, use_container_width=True)
        promo = subset.assign(promotion=subset.onpromotion.gt(0).map({True: "Có", False: "Không"}))
        fig = px.box(promo, x="promotion", y="sales", title="Khuyến mãi và sales · mô tả tương quan")
        fig.update_layout(xaxis_title="Có khuyến mãi", yaxis_title="sales")
        st.plotly_chart(fig, use_container_width=True)
    quality_path = result_dir / "metrics/data_quality.json"
    if quality_path.exists():
        with st.expander("Chất lượng CSV gốc trước xử lý · từ lần train"):
            st.json(json.loads(quality_path.read_text(encoding="utf-8")))
    st.caption("Ngày lễ: chạy src.eda với --holidays và --stores khi có CSV bổ sung; biểu đồ HTML nằm trong results/<nguồn>/figures/.")
    show_table(subset)
elif page in ["Dự báo nhu cầu", "Quản lý tồn kho"]:
    if artifact is None:
        st.info(f"Chưa có mô hình. Chạy `python -m src.train --data {data_path} --output results/{run_name} --model-dir models/{run_name} --quick`.")
    else:
        available = data.merge(pd.DataFrame(artifact["series"], columns=KEYS), on=KEYS, how="inner")
        if available.empty:
            st.error("CSV không có chuỗi nào đã được huấn luyện.")
        else:
            history = selection(available, "forecast" if page == "Dự báo nhu cầu" else "inventory")
            horizon = st.selectbox("Số ngày dự báo", [7, 30])
            if page == "Dự báo nhu cầu":
                identity = (str(history.date.max()), int(history.store_nbr.iloc[0]), history.family.iloc[0],
                            horizon, data_path, str(absolute.stat().st_mtime_ns), str(model_path.stat().st_mtime_ns))
                if st.button("Thực hiện dự báo", type="primary"):
                    try:
                        with st.spinner("Đang tính dự báo từ mô hình đã huấn luyện…"), threadpool_limits(limits=1):
                            st.session_state["forecast_result"] = (identity, forecast(artifact, history, horizon))
                    except ValueError as exc:
                        st.error(str(exc))
                saved = st.session_state.get("forecast_result")
                if saved is not None and saved[0] == identity:
                    result = saved[1]
                    st.metric("Tổng nhu cầu dự báo", f"{result.prediction.sum():,.2f}")
                    fig = go.Figure()
                    past = history.tail(60)
                    fig.add_scatter(x=past.date, y=past.sales, name="Lịch sử", line_color="#94a3b8")
                    fig.add_scatter(x=result.date, y=result.prediction, name="Dự báo", line_color="#2563eb")
                    fig.update_layout(title="Nhu cầu lịch sử và dự báo")
                    format_chart(fig)
                    show_table(result)
                    st.download_button("Tải kết quả CSV", result.to_csv(index=False).encode("utf-8-sig"), "du_bao_nhu_cau.csv", "text/csv")
                st.caption("Dự báo từ ngày sau quan sát cuối của chuỗi. Lag tương lai dùng dự báo trước đó; không dùng biến ngoại sinh tương lai.")
            else:
                st.info("Tồn kho do người dùng nhập. Chỉ tính khi nhu cầu và tồn kho cùng một đơn vị đã quy đổi.")
                with st.form("inventory_form"):
                    col1, col2 = st.columns(2)
                    current = col1.number_input("Tồn kho hiện tại", min_value=0.0, value=300.0)
                    lead = col2.number_input("Thời gian giao hàng (ngày)", min_value=1, max_value=365, value=5)
                    safety = col1.number_input("Tồn kho an toàn", min_value=0.0, value=50.0)
                    incoming = col2.number_input("Hàng chắc chắn về trong thời gian giao hàng", min_value=0.0, value=0.0)
                    same_unit = st.checkbox("Tôi xác nhận tồn kho và dự báo cùng đơn vị; hàng đang về có thể dùng đúng hạn")
                    submit = st.form_submit_button("Tính đề xuất nhập hàng", type="primary")
                if submit:
                    try:
                        if lead > horizon:
                            raise ValueError("Lead time vượt chân trời dự báo. Chọn 30 ngày hoặc dùng CLI dự báo dài hơn.")
                        with threadpool_limits(limits=1):
                            result = forecast(artifact, history, horizon)
                        decision = recommend_order(result.prediction, current, int(lead), safety, incoming, same_unit)
                        cols = st.columns(3)
                        cols[0].metric("Nhu cầu trong lead time", f"{decision['lead_time_demand']:,.2f}")
                        cols[1].metric("Tồn kho mục tiêu", f"{decision['target_stock']:,.2f}")
                        cols[2].metric("Lượng cần nhập thêm", f"{decision['order_quantity']:,.2f}")
                        if decision["shortage_risk"]:
                            st.warning(decision["status"])
                        elif decision["excess_risk"]:
                            st.warning(decision["status"])
                        else:
                            st.success(decision["status"])
                        st.caption("Hàng mới đặt có thể chỉ đến cuối lead time; đề xuất này không đảm bảo tránh mọi thiếu hàng trước khi nhận hàng.")
                    except ValueError as exc:
                        st.error(str(exc))
                st.markdown("**Công thức:** D = tổng dự báo trong lead time; A = tồn hiện tại + hàng chắc chắn về; T = D + tồn an toàn; Q = max(0, T − A).")
                st.caption("Cảnh báo thiếu khi A < T; dư khi A > 2 × T. Lượng nhập giữ số thập phân theo đơn vị gốc. Chỉ làm tròn lên khi đơn vị hàng là rời rạc. Chưa có hàm mục tiêu chi phí.")
elif page == "Đánh giá mô hình":
    if manifest is None:
        st.info("Chưa có kết quả thực nghiệm. Chạy huấn luyện theo README.")
    else:
        st.subheader(f"Mô hình được chọn: {manifest['selected_model']}")
        st.write(manifest["selection_reason"])
        table = pd.read_csv(result_dir / "metrics/comparison.csv")
        show_table(table)
        st.caption("MAE: sai lệch tuyệt đối trung bình. RMSE: nhấn mạnh lỗi lớn. WAPE (%): tổng sai số tuyệt đối / tổng |thực tế| × 100; không xác định khi mẫu số = 0. Thấp hơn tốt hơn.")
        st.caption("Thời gian train trong bảng là lần refit train + validation để test. Thời gian suy luận gồm xử lý CSV trong bộ nhớ, đặc trưng và toàn bộ dự báo đệ quy; không bao gồm tải file/model.")
        pred = pd.read_csv(result_dir / "metrics/test_predictions.csv", parse_dates=["date"])
        totals = pred.groupby("date", as_index=False)[["sales", "prediction"]].sum().rename(columns={"sales": "Thực tế", "prediction": "Dự báo"})
        format_chart(px.line(totals, x="date", y=["Thực tế", "Dự báo"], title="Test giữ riêng · tổng các chuỗi"))
        st.subheader("Trước và sau tìm kiếm siêu tham số")
        tuning = pd.read_csv(result_dir / "metrics/tuning.csv")
        tuning["stage"] = tuning.stage.map({"before_tuning": "Cấu hình ban đầu", "random_search": "Tìm kiếm ngẫu nhiên"})
        show_table(tuning.rename(columns={"stage": "Giai đoạn", "trial": "Lượt", "prediction_rows": "Số dòng dự báo"}))
        with st.expander("Cấu hình, mốc chia dữ liệu và môi trường thực nghiệm"):
            st.json(manifest)
        st.caption("Test dùng evaluation_model.joblib fit đến trước test. model.joblib phục vụ dự báo mới được refit trên toàn bộ lịch sử sau đánh giá. Không dùng model triển khai để tái tính test.")
        st.subheader("Bằng chứng quá trình huấn luyện")
        st.write("Mỗi cấu hình được fit trên tập train, sau đó dự báo đệ quy trên kiểm định. "
                 "Thời điểm chọn mô hình được ghi trước khi đánh giá tập kiểm tra.")
        history_path = result_dir / "metrics/training_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path)
            if not history.empty:
                history["Cấu hình"] = history.trial.map(lambda t: "Gốc" if t == 0 else f"Tuning {t}")
                fig = px.line(history, x="iteration", y="RMSE", color="Cấu hình",
                              title="HGB: RMSE trên mẫu train theo vòng boosting",
                              labels={"iteration": "Vòng boosting", "RMSE": "RMSE trên mẫu train"})
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Chỉ số in-sample trên tối đa 5.000 hàng train cố định, dùng quan sát quá trình học; "
                           "không thay MAE/RMSE/WAPE kiểm định hoặc kiểm tra. Ridge và Random Forest không cần epoch như mạng nơ-ron.")
        for filename, label in [("training.log", "Tải nhật ký huấn luyện"),
                                ("training_events.jsonl", "Tải dấu vết từng bước train")]:
            log_path = result_dir / filename
            if log_path.exists():
                st.download_button(label, log_path.read_bytes(), filename, "text/plain")
        if (result_dir / "training.log").exists():
            with st.expander("Xem các bước thực thi trong nhật ký"):
                st.code((result_dir / "training.log").read_text(encoding="utf-8"), language="text")
