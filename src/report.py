"""Sinh phần kết quả báo cáo từ artefact thực thi, không tự điền số liệu giả."""
import argparse
import json
from pathlib import Path
import pandas as pd


def markdown_table(frame):
    rows = [[str(col) for col in frame.columns], ["---"] * len(frame.columns)]
    for row in frame.itertuples(index=False, name=None):
        rows.append([f"{x:.4f}" if isinstance(x, float) else str(x).replace("|", "/") for x in row])
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def generate_report(results="results/demo", output="docs/bao_cao_giua_ky.md"):
    root = Path(results)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    quality = json.loads((root / "metrics/data_quality.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(root / "metrics/comparison.csv")
    summary = pd.read_csv(root / "metrics/tuning_summary.csv")
    selected = comparison[comparison.selected].iloc[0]
    table = markdown_table(comparison[["model", "validation_WAPE_pct", "test_MAE", "test_RMSE", "test_WAPE_pct", "train_seconds", "inference_seconds"]])
    splits = markdown_table(pd.DataFrame([{"Tập": k, **v} for k, v in manifest["split"].items()]))
    source = manifest["metadata"]
    simulated = source.get("simulated", False)
    dataset_note = (
        "Workspace chưa được cung cấp CSV Kaggle ở thời điểm triển khai. Lần chạy này dùng mô phỏng, "
        "không công bố như kết quả Kaggle."
        if simulated else
        "Lần chạy này dùng CSV người dùng cung cấp. Nguồn được công bố theo metadata người dùng; "
        "nhóm cần xác nhận đã tải đúng dữ liệu gốc và tuân thủ điều khoản sử dụng."
    )
    limitation_note = (
        "Chưa thực nghiệm trên Kaggle chính thức vì thiếu CSV. Dữ liệu mô phỏng có quy luật đơn giản; "
        "kết quả không suy rộng sang bán lẻ thật."
        if simulated else
        "Kết quả chỉ đại diện cho CSV và các chuỗi được chọn trong lần chạy này; "
        "không mặc định suy rộng cho toàn bộ chuỗi hoặc thời kỳ khác."
    )
    conclusion_note = (
        "Chất lượng trên Kaggle cần được đo sau khi có dataset chính thức."
        if simulated else
        "Cần bổ sung đánh giá trên nhiều mốc thời gian độc lập trước khi kết luận chất lượng vận hành."
    )
    report = f"""# BÁO CÁO GIỮA KỲ

**Đề tài:** Xây dựng hệ thống dự báo nhu cầu hàng hóa hỗ trợ tối ưu quản lý tồn kho bằng Machine Learning.

Người thực hiện: [điền danh sách thành viên]. Giảng viên: [điền thông tin xác nhận].
Bảng nhiệm vụ: [phan_cong_nhiem_vu.md](phan_cong_nhiem_vu.md).
Kết quả dưới đây được sinh trực tiếp bằng `python -m src.report --results {root}`.
Thời điểm thực nghiệm UTC: {manifest['created_utc']}.

## 1. Giới thiệu đề tài

Nghiên cứu xây dựng một pipeline dự báo số lượng bán theo ngày từ lịch sử bán hàng,
đồng thời cung cấp chức năng hỗ trợ lập kế hoạch nhập hàng. Hệ thống chạy trên máy cá nhân,
tách quá trình huấn luyện khỏi suy luận và giao diện Streamlit. Bài toán chính sử dụng các
thuật toán học máy truyền thống, không gọi API mô hình ngôn ngữ lớn.

## 2. Lý do lựa chọn đề tài

Nhu cầu biến động theo thời gian làm tăng rủi ro thiếu hàng hoặc dư tồn. Dự báo giúp cung cấp
một ước lượng định lượng cho người quản lý. Dữ liệu dạng bảng theo thời gian phù hợp cho việc
so sánh mô hình tuyến tính và các mô hình cây, đồng thời làm rõ các vấn đề rò rỉ dữ liệu thường gặp.

## 3. Mục tiêu nghiên cứu

Xây dựng tiền xử lý nhất quán, tạo đặc trưng chỉ từ quá khứ, huấn luyện ít nhất hai thuật toán,
tìm kiếm siêu tham số, lựa chọn mô hình bằng validation và đánh giá trên test độc lập.
Sản phẩm gồm mã nguồn, mô hình đã lưu, kết quả đo thực, giao diện dự báo và quy tắc nhập hàng.

## 4. Phát biểu bài toán

Với mỗi chuỗi s = (cửa hàng, nhóm hàng), tại cuối ngày t, dự báo sales cho t+1 đến t+H,
H = 7 hoặc 30 trên giao diện. CLI cho phép 1–365 ngày. Đây là dự báo sales ở cấp nhóm hàng,
không phải cấp SKU. Sales quan sát là đại diện cho nhu cầu; không quan sát được nhu cầu bị mất
do hết hàng. Không diễn giải sales thành doanh thu tiền tệ.

## 5. Giới thiệu dataset

Nguồn lần chạy này: **{source['source']}**.
Đánh dấu mô phỏng: **{source.get('simulated', False)}**.
CSV đã kiểm tra thực tế có {quality['rows']:,} dòng; dùng {quality['selected_rows']:,} dòng
cho {source['series_count']} chuỗi. Cột thực tế: {', '.join(quality['columns'])}.
Đơn vị: {source['unit']}.

Bộ dữ liệu chính thức dự kiến là [Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
Tài liệu nguồn xác nhận train.csv gồm cửa hàng, nhóm hàng, sales và onpromotion;
sales có thể có phần thập phân. {dataset_note}
Các tệp chính thức cần tải sau khi đăng nhập
và chấp thuận điều khoản; README có lệnh huấn luyện riêng, tách artefact demo/official.

## 6. Phân tích và tiền xử lý dữ liệu

Thiếu theo cột trước xử lý: `{json.dumps(quality['missing_values'], ensure_ascii=False)}`.
Số dòng trùng: {quality['duplicate_rows']}; trùng khóa: {quality['duplicate_keys']}.
Sales trung bình: {quality['sales_summary']['mean']:.4f}, nhỏ nhất:
{quality['sales_summary']['min']:.4f}, lớn nhất: {quality['sales_summary']['max']:.4f}.

Chương trình từ chối ngày không hợp lệ, khóa trùng, sales thiếu/âm/vô hạn, family rỗng và
mã cửa hàng không nguyên dương. Thiếu onpromotion được điền 0 cho EDA; biến này không được dùng
trong mô hình nên không biến giả định điền thành thông tin dự báo.
Ngày thiếu trong mỗi chuỗi được chèn và forward fill từ quá khứ để lag có nghĩa theo ngày lịch.
Ngày chèn mang is_observed=False, tuyệt đối không làm nhãn train/test. Lần chạy này chèn
{quality['inserted_calendar_days']} ngày và loại {quality['excluded_warmup_observed_rows']} quan sát warm-up
không đủ 28 ngày lịch sử. Không dùng backward fill.

EDA HTML trong `{root}/figures/eda_*.html` gồm xu hướng ngày, tổng theo cửa hàng/nhóm hàng,
trung bình theo thứ/tháng và phân phối khi có/không khuyến mãi. Biểu đồ mô tả tương quan,
không chứng minh khuyến mãi gây tăng nhu cầu. Biểu đồ lễ chỉ sinh khi thực sự cung cấp
holidays_events.csv; nếu có stores.csv thì ghép National/Regional/Local đúng địa bàn,
loại ngày transferred và Work Day. Không tự gán lịch lễ ngoài dữ liệu.

## 7. Phương pháp Machine Learning

- **Ridge:** hồi quy tuyến tính với phạt L2, làm baseline dễ giải thích.
- **Random Forest:** tổng hợp nhiều cây để học tương tác phi tuyến, hạn chế overfit bằng độ sâu và min leaf.
- **HistGradientBoosting:** boosting cây theo histogram tích hợp trong scikit-learn; dùng thay LightGBM/XGBoost
  để giảm phụ thuộc cài đặt trên máy cá nhân. Đây vẫn là thuật toán khác Random Forest và Ridge.
- **Seasonal naive:** lấy sales của 7 ngày trước, sau 7 bước dùng lại dự báo đã sinh; mốc đối chiếu không train.

Đặc trưng: ngày, tháng, thứ, cuối tuần, chu kỳ sin/cos thứ và ngày trong năm, trend theo ngày;
lag 1/7/14/28; rolling mean và std(ddof=0) của 7/14/28 ngày. Luôn shift(1) trước rolling.
Mã cửa hàng và nhóm hàng được one-hot; numeric được chuẩn hóa. Transformer chỉ fit trên phần train
của từng giai đoạn. Không dùng onpromotion, oil, transactions hay sự kiện tương lai trong mô hình
hiện tại; không cần giả định giá trị tương lai của các biến này. Không có API LLM.

## 8. Quy trình huấn luyện

Chia hoàn toàn theo ngày, không random split và không để cùng ngày rơi vào nhiều tập:

{splits}

Các mô hình có cùng tập quan sát train và cùng origin, horizon validation/test. Validation được
dự báo đệ quy từ cuối train; test từ cuối train+validation. Không nạp sales thật của ngày trước
trong holdout để tạo lag cho ngày kế tiếp. Các dự báo âm được chặn tại 0 cho mọi mô hình.
Mô hình tốt nhất được khóa trước khi chạy test. Test có ground truth được tách từ lịch sử;
test.csv Kaggle không có sales và không dùng làm nhãn đánh giá.

## 9. Tối ưu siêu tham số

Sử dụng sklearn ParameterSampler, random_state=42, quick={manifest['quick']},
với số lượt cụ thể ghi trong `{root}/metrics/tuning.json`.
Mỗi cấu hình fit trên train và chấm bằng recursive validation một origin (không phải random K-fold).
Không gọi là RandomizedSearchCV: vòng tìm kiếm tự viết để bảo toàn semantics dự báo nhiều bước.
Không gian tìm kiếm: Ridge alpha 0.1/1/10/100/1000;
RF depth 6/10/16/None, min leaf 1/3/6, max features 0.7/1.0;
HGB learning rate 0.03/0.06/0.12, max leaves 7/15/31, L2 0/1/10.
Cấu hình mặc định cũng được chấm và có thể thắng; tuning không được giả định luôn cải thiện.
Toàn bộ kết quả và thời gian từng lượt nằm trong tuning.json và tuning.csv.

{markdown_table(summary)}

## 10. Kết quả thực nghiệm và so sánh

**Chỉ có giá trị với nguồn dữ liệu ghi ở mục 5.** Thời gian đo bằng perf_counter trên
{manifest['environment']['platform']}, Python {manifest['environment']['python']},
scikit-learn {manifest['environment']['scikit-learn']}, NumPy {manifest['environment']['numpy']}.
Native thread pool và RF n_jobs giới hạn 1 để phù hợp laptop. Timing phụ thuộc phần cứng và tải máy.

{table}

MAE = mean(|y−ŷ|), RMSE = sqrt(mean((y−ŷ)²)), cùng đơn vị sales.
WAPE = 100 × sum(|y−ŷ|)/sum(|y|); mẫu số 0 thì trả None/không xác định.
WAPE toàn cục ưu tiên chuỗi có quy mô sales lớn, không phải trung bình phần trăm từng chuỗi.
`train_seconds` là refit trên train+validation để đánh giá test, không gồm tổng tuning;
`inference_seconds` gồm validation trong bộ nhớ, tạo đặc trưng và toàn bộ đệ quy test,
không gồm đọc đĩa/nạp model. Tổng lần chạy: {manifest['total_seconds']:.3f} giây.

Mô hình được chọn: **{manifest['selected_model']}**. {manifest['selection_reason']}
Tham số: `{json.dumps(manifest['best_params'], ensure_ascii=False)}`.
Test MAE {selected.test_MAE:.4f}, RMSE {selected.test_RMSE:.4f}, WAPE {selected.test_WAPE_pct:.4f}%.
Đây là kết quả đo, không phải cam kết độ chính xác khi vận hành.
Biểu đồ test: `{root}/figures/test_actual_vs_forecast.html`; sai số từng chuỗi: metrics/test_by_series.csv.

Artefact `evaluation_model.joblib` chỉ fit đến trước test để tái hiện đánh giá. Sau khi hoàn tất
đánh giá độc lập, `model.joblib` được refit trên toàn bộ lịch sử để dự báo mới. Không dùng bản
refit này để báo lại test. SHA-256 của CSV và danh sách chuỗi nằm trong run_manifest.json.

## 11. Mô tả hệ thống demo

Streamlit gồm sáu trang: Tổng quan, Phân tích dữ liệu, Dự báo nhu cầu, Quản lý tồn kho,
Đánh giá mô hình, Giới thiệu hệ thống. Người dùng lọc cửa hàng/nhóm hàng, chọn 7 hoặc 30 ngày,
xem bảng/biểu đồ và tải CSV. Giao diện chỉ nạp model đã train, không tự kích hoạt huấn luyện.
Nguồn mô phỏng được hiển thị rõ. Đầu vào thiếu model hoặc CSV sai được thông báo.
Pipeline CLI `src.predict` nhận CSV lịch sử mới có sales đã biết, tiền xử lý như train,
nạp artefact và dự báo từ ngày sau ngày chốt cuối. Không hỗ trợ chuỗi cửa hàng/nhóm hàng chưa học.

## 12. Ứng dụng dự báo trong quản lý tồn kho

Người dùng nhập tồn hiện tại C, hàng chắc chắn về I, lead time L và tồn an toàn S.
Sau khi xác nhận cùng đơn vị: D_L = tổng dự báo L ngày; A = C+I; T = D_L+S;
Q = max(0,T−A). Cảnh báo thiếu nếu A<T, dư nếu A>2T (ngưỡng quy tắc, không học từ dữ liệu).
Lead time vượt horizon bị từ chối. Hàng đang về giả định đến đủ sớm để sử dụng;
Q đặt hôm nay không đảm bảo tránh thiếu hàng trước khi đơn mới về. Đơn vị rời rạc có thể làm tròn Q lên,
còn đơn vị liên tục giữ phần thập phân. Ví dụ SRS 40/45/50/45/50, C=120, I=20, S=30 cho Q=120
chỉ là ví dụ kiểm thử công thức, không phải kết quả dự báo thật.
Không khẳng định tối thiểu hóa chi phí; chưa có giá nhập, chi phí giữ hàng, chi phí thiếu hàng,
MOQ, lịch giao hàng hoặc hàm mục tiêu. Không tự động tạo đơn mua.

## 13. Hạn chế

{limitation_note} Chỉ một origin validation và một test,
horizon test của lần chạy là {manifest['horizon']} ngày; chức năng 7/30 ngày không đồng nghĩa
đã đánh giá độc lập cả hai horizon. Chưa có khoảng dự báo hay backtest nhiều mùa.
Không dùng biến ngoại sinh nên có thể bỏ lỡ lễ/khuyến mãi/biến động kinh tế.
Forward fill ngày thiếu là giả định giữ nhu cầu gần nhất, có thể gây sai số; chỉ là state, không là nhãn.
Mức tổng hợp nhóm hàng có thể không đồng nhất đơn vị vật lý; muốn nhập hàng theo SKU cần dữ liệu chi tiết.
Quick mode tối đa 12 chuỗi nếu không ghi --max-series; kết quả chỉ đại diện chuỗi được chọn.

## 14. Hướng phát triển

Thu thập CSV chính thức, đối chiếu đơn vị và bổ sung rolling-origin backtest; thử LightGBM/XGBoost
trong cùng protocol; kết hợp lịch lễ theo địa bàn và kế hoạch khuyến mãi đã biết;
đánh giá theo từng nhóm hàng/horizon; bổ sung khoảng dự báo và theo dõi drift.
Để tối ưu tồn kho thực sự, cần dữ liệu tồn theo SKU, phân phối lead time, chi phí và ràng buộc vận hành.

## 15. Kết luận

Sản phẩm thực hiện đủ pipeline train–tuning–validation–test–lưu/nạp–suy luận,
so sánh ba thuật toán học máy và một mốc naive trên dữ liệu hiện có, có giao diện và quy tắc tồn kho.
Giá trị của bản giữa kỳ là quy trình tái lập, tránh leakage và công bố giới hạn dữ liệu.
{conclusion_note}
Kết quả kiểm thử thực thi ghi riêng trong [kiem_thu.md](kiem_thu.md) và JUnit XML;
không suy ra test đã qua chỉ từ việc sinh báo cáo này.

## 16. Tài liệu tham khảo

1. Tài liệu giảng viên “BÀI TẬP LỚN GIỮA KỲ”, Giua ky.docx do người dùng cung cấp,
   mục 1–2: yêu cầu huấn luyện, so sánh, tuning, pipeline, metric/timing và bốn thành phần bài nộp.
2. Requirement_Specification.docx trong workspace: đặc tả nhóm, đặc biệt quy tắc tồn kho mục 6.5.
3. [Kaggle, Store Sales – Dataset description](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
4. [scikit-learn, ParameterSampler](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterSampler.html).
5. [scikit-learn, Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).
6. [scikit-learn, RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).
7. [scikit-learn, HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
8. [Streamlit, App testing](https://docs.streamlit.io/develop/api-reference/app-testing).

Tài liệu web mô tả thuật toán/schema; API thực thi được cố định theo requirements.txt và kiểm thử trên
phiên bản trong manifest. Không trích dẫn nguồn giả hoặc điền kết quả Kaggle chưa chạy.
"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results/demo")
    p.add_argument("--output", default="docs/bao_cao_giua_ky.md")
    args = p.parse_args()
    print(generate_report(args.results, args.output))
