# Dự báo nhu cầu hàng hóa và hỗ trợ quản lý tồn kho

Ứng dụng Machine Learning chạy cục bộ: kiểm tra dữ liệu → EDA → tạo đặc trưng quá khứ →
huấn luyện và tuning → đánh giá test → dự báo → đề xuất nhập hàng. Giao diện Streamlit có
sáu trang tiếng Việt. Các mô hình dự báo dùng scikit-learn.

Dữ liệu [Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
gồm **3.000.888 dòng, 54 cửa hàng, 33 nhóm hàng**, từ 01/01/2013 đến 15/08/2017.
EDA dùng toàn bộ CSV. Thực nghiệm chính thức dùng **54 cửa hàng × 3 nhóm**
`BEVERAGES`, `GROCERY I`, `PRODUCE`: **162 chuỗi, 272.808 quan sát**.
Phạm vi này được chọn trước khi xem metric để phù hợp máy cá nhân; chưa train đủ 33 nhóm.

## 1. Cài đặt môi trường

Mở terminal tại thư mục gốc dự án. Khuyến nghị Python 3.11; môi trường thực nghiệm đã kiểm tra
là **macOS arm64, Python 3.9.6**. Phiên bản thư viện nằm trong `requirements.txt`;
`requirements-lock.txt` lưu môi trường thực nghiệm để đối chiếu.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

Nếu PowerShell chặn kích hoạt, dùng `.\.venv\Scripts\python.exe` thay cho `python` trong các lệnh sau.
Cần mạng để cài thư viện và tải dữ liệu. Windows/Linux/Python 3.11 chưa được kiểm thử thực tế.

## 2. Chuẩn bị dữ liệu

Đăng nhập Kaggle, chấp thuận điều khoản cuộc thi, tải và giải nén các file vào:

```text
data/raw/train.csv             Bắt buộc: lịch sử bán hàng có nhãn sales
data/raw/stores.csv            Tùy chọn: địa bàn cửa hàng cho EDA ngày lễ
data/raw/holidays_events.csv    Tùy chọn: lịch lễ/sự kiện cho EDA
data/raw/train.metadata.json    Tùy chọn: nguồn và đơn vị dữ liệu đã xác nhận
```

Workspace hiện đã có ba CSV trên. Khi sao chép/clone dự án sang máy khác, cần chuẩn bị lại
vì `data/raw/*.csv` không được lưu trong Git. Xem [hướng dẫn dữ liệu](data/raw/README.md).
Schema bắt buộc: `date,store_nbr,family,sales`; `onpromotion` chỉ dùng cho EDA.
Kaggle `test.csv` không có `sales`, không dùng làm đầu vào lịch sử cho pipeline này.

## 3. Chạy toàn bộ chương trình từ đầu

Sau khi cài đặt, chuẩn bị CSV và kích hoạt `.venv`, chạy lần lượt các lệnh dưới đây tại thư mục gốc.
Đây là **toàn bộ quy trình với phạm vi thực nghiệm chính thức 162 chuỗi**, không dùng `--quick`.
`src.train` tự tạo đặc trưng, chia train/validation/test, train ba thuật toán, tuning, chọn model,
đánh giá và lưu cả model đánh giá lẫn model triển khai.

```bash
# 1. Kiểm tra schema và chất lượng dữ liệu
python -m src.data_processing --data data/raw/train.csv

# 2. Phân tích toàn bộ dữ liệu và xuất biểu đồ HTML
python -m src.eda --data data/raw/train.csv --output results/official --holidays data/raw/holidays_events.csv --stores data/raw/stores.csv

# 3. Huấn luyện, tuning và đánh giá trên test giữ riêng
python -m src.train --data data/raw/train.csv --output results/official --model-dir models/official --horizon 30 --trials 3 --families BEVERAGES "GROCERY I" PRODUCE

# 4. Tính lại metric từ CSV dự báo test đã lưu
python -m src.evaluate --predictions results/official/metrics/test_predictions.csv --output results/official/metrics/recomputed_test.json

# 5. Nạp lại model để tái hiện test; xuất dự báo 7/30 ngày cho toàn bộ chuỗi đã train
python -m src.verify_run --results results/official

# 6. Dự báo riêng cửa hàng 1 / BEVERAGES, kèm metadata nguồn
python -m src.predict --input data/raw/train.csv --model models/official/model.joblib --store 1 --family BEVERAGES --horizon 7 --output results/official/forecast_7_days.csv
python -m src.predict --input data/raw/train.csv --model models/official/model.joblib --store 1 --family BEVERAGES --horizon 30 --output results/official/forecast_30_days.csv

# 7. Sinh báo cáo Markdown từ kết quả thực nghiệm
python -m src.report --results results/official --output docs/bao_cao_giua_ky.md

# 8. Chạy bộ kiểm thử tự động
python -m pytest -q --junitxml=results/metrics/pytest.xml

# 9. Mở giao diện
python -m streamlit run app.py
```

Nếu chỉ có `train.csv`, thay lệnh EDA ở bước 2 bằng:

```bash
python -m src.eda --data data/raw/train.csv --output results/official
```

Mở **http://localhost:8501** và chọn nguồn **CSV người dùng / Kaggle** nếu cần.
Trên trang **Dự báo nhu cầu**, chọn cửa hàng/nhóm hàng đã train và nhấn **Thực hiện dự báo**.
Trang **Quản lý tồn kho** dùng dự báo cùng các giá trị tồn kho người dùng nhập để đề xuất nhập hàng.
Dừng server bằng `Ctrl+C`. Giao diện chỉ nạp model đã train; huấn luyện chạy bằng CLI.

Các lệnh trên ghi vào `results/official`, `models/official` và sinh lại báo cáo Markdown.
Đổi `--output`/`--model-dir` nếu cần giữ riêng lần chạy; các bước sau phải trỏ tới đúng thư mục mới.
Model `.joblib` không được lưu trong Git, cần train lại khi chưa có model trên máy.
Dữ liệu kết thúc 15/08/2017 nên dự báo bắt đầu **16/08/2017**.

### Mở lại khi đã có model và kết quả

macOS / Linux:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

### Huấn luyện toàn bộ 33 nhóm hàng (tùy chọn)

Bỏ các bộ lọc `--stores`, `--families`, `--max-series` và không dùng `--quick` để học cả
**54 × 33 = 1.782 chuỗi**. Lệnh sau lưu riêng; thời gian và RAM cho phạm vi này chưa được đo:

```bash
python -m src.train --data data/raw/train.csv --output results/full_dataset --model-dir models/full_dataset --horizon 30 --trials 3
```

Để đánh giá/kiểm chứng lần này, dùng `--results results/full_dataset` hoặc thay đường dẫn
`results/official`/`models/official` tương ứng ở các lệnh trên; xuất báo cáo ra file riêng.
Giao diện nguồn Kaggle hiện đọc `results/official`, nên vẫn hiển thị lần chạy chính thức.

## 4. Đầu ra và kết quả tham chiếu

| Đường dẫn | Nội dung |
| --- | --- |
| `models/official/evaluation_model.joblib` | Model fit train + validation để tái hiện test |
| `models/official/model.joblib` | Model refit toàn lịch sử sau đánh giá để dự báo mới |
| `results/official/run_manifest.json` | Nguồn, SHA-256, chuỗi đã train, mốc chia và môi trường |
| `results/official/metrics/comparison.csv` | Metric validation/test và thời gian các thuật toán |
| `results/official/metrics/tuning.csv`, `tuning_summary.csv` | Từng cấu hình và so sánh trước/sau tuning |
| `results/official/metrics/verification.json` | Kiểm chứng model nạp lại và dự báo 7/30 ngày |
| `results/official/forecast_{7,30}_days_all_series.csv` | Dự báo toàn bộ 162 chuỗi do `src.verify_run` xuất |
| `results/official/forecast_{7,30}_days.csv` | Dự báo cửa hàng 1 / BEVERAGES do `src.predict` xuất |
| `results/official/figures/` | Biểu đồ EDA, chia thời gian, train, tuning và test; mở HTML bằng trình duyệt |
| `results/official/training.log`, `training_events.jsonl`, `source_snapshot/` | Nhật ký huấn luyện và mã nguồn tại thời điểm chạy |
| `docs/bao_cao_giua_ky.md` | Báo cáo sinh từ kết quả thực nghiệm |
| `results/metrics/pytest.xml` | Kết quả bộ kiểm thử tự động |

Kết quả tham chiếu trên 4.860 quan sát test, lấy từ `results/official/metrics/comparison.csv`.
**Random Forest** được chọn theo validation.

| Thuật toán | MAE test | RMSE test | WAPE test (%) |
| --- | ---: | ---: | ---: |
| Seasonal naive | 517,555 | 899,231 | 14,986 |
| Ridge | 542,383 | 797,537 | 15,705 |
| Random Forest | 445,633 | 692,496 | 12,904 |
| HistGradientBoosting | 476,471 | 732,513 | 13,797 |

MAE/RMSE/WAPE càng thấp càng tốt; không diễn giải `100 − WAPE` là accuracy.
Thời gian train trong bảng comparison là lần refit train + validation, không gồm tổng tuning.
Suy luận đo xử lý dữ liệu trong bộ nhớ và cả horizon, không gồm đọc CSV/nạp model;
ms/dòng là thời gian batch chia số dòng.

## 5. Quy tắc mô hình và dữ liệu mới

- Chia theo ngày: 30 ngày validation, 30 ngày test, phần trước là train. Mặc định cần ít nhất
  `60 + 2 × horizon` ngày và đủ lịch sử cho từng chuỗi; không random split.
- Lag 1/7/14/28, rolling mean/std 7/14/28 với `shift(1)`, đặc trưng lịch và chu kỳ.
  Encoder/scaler chỉ fit trên tập đang dùng để học. Ngày thiếu được forward fill làm state,
  đánh dấu `is_observed=False` và không dùng làm nhãn train/test.
- Ba mô hình: Ridge, Random Forest, HistGradientBoosting; seasonal naive dùng lag_7 làm mốc.
  Tuning bằng ParameterSampler và recursive validation; seed 42.
- Chọn WAPE validation thấp nhất, fallback MAE nếu mẫu số bằng 0; khóa lựa chọn trước test.
  Dự báo đệ quy dùng dự báo đã sinh để cập nhật lag, không dùng sales thật trong holdout.
- Khuyến mãi/ngày lễ chỉ dùng cho EDA; mô hình hiện chưa dùng biến ngoại sinh tương lai.

Để dự báo từ lịch sử cập nhật, đặt CSV tại `data/raw/history.csv`, có cùng schema và ít nhất
**28 ngày lịch sử mỗi chuỗi**. Chọn chuỗi đã xuất hiện khi train; dự báo bắt đầu sau ngày cuối đầu vào:

```bash
python -m src.predict --input data/raw/history.csv --model models/official/model.joblib --store 1 --family BEVERAGES --horizon 30 --output results/official/forecast_new.csv
```

Bỏ `--store`/`--family` chỉ khi mọi chuỗi trong CSV đều đã được train và cập nhật cùng ngày cuối.
Đầu ra gồm `date,store_nbr,family,prediction` và sidecar `.metadata.json` ghi nguồn, hash và origin.
Chỉ nạp `.joblib` tin cậy do nhóm tự train.

## 6. Giao diện và tồn kho

Sáu trang gồm **Tổng quan**, **Phân tích dữ liệu**, **Dự báo nhu cầu**, **Quản lý tồn kho**,
**Đánh giá mô hình**, **Giới thiệu**. Giao diện hỗ trợ dự báo 7/30 ngày, tải CSV,
xem metric/tuning và nhập dữ liệu tồn kho.

```text
D_L = tổng dự báo trong L ngày giao hàng
A   = tồn hiện tại + hàng chắc chắn về và sử dụng đúng hạn
T   = D_L + tồn an toàn
Q   = max(0, T − A)
```

Cần xác nhận tồn kho và dự báo cùng đơn vị. Sales tổng nhóm có thể có phần thập phân,
không tự chuyển thành số SKU/thùng/tiền. Lead time phải nằm trong horizon.
Thiếu khi `A < T`, dư theo quy tắc khi `A > 2 × T`; đây là hỗ trợ quyết định,
chưa tối ưu chi phí hoặc tự động đặt hàng.

## 7. Tài liệu và xử lý lỗi

Mã nguồn pipeline ở `src/`, giao diện ở `app.py`, kiểm thử ở `tests/`.
Đọc [báo cáo](docs/bao_cao_giua_ky.md), [minh chứng huấn luyện](docs/minh_chung_huan_luyen.md),
[kiểm thử](docs/kiem_thu.md), [đối chiếu yêu cầu](docs/doi_chieu_yeu_cau.md),
[thuyết trình](docs/huong_dan_thuyet_trinh.md), [phân công](docs/phan_cong_nhiem_vu.md)
và [nhật ký xây dựng](docs/nhat_ky_build_du_an.md).

| Hiện tượng | Cách xử lý |
| --- | --- |
| `ModuleNotFoundError` | Kích hoạt đúng `.venv`, cài `requirements.txt` |
| Chưa có model | Chạy bước train với đúng `results/official` và `models/official` |
| Thiếu `sales` | Dùng lịch sử có nhãn, không dùng Kaggle `test.csv` |
| Ngày/khóa/sales không hợp lệ | Sửa dữ liệu tại nguồn theo thông báo kiểm tra |
| Không đủ ngày chia tập | Bổ sung lịch sử thật hoặc chọn chuỗi đủ dài |
| Cửa hàng/nhóm chưa train | Chọn chuỗi trong manifest hoặc train lại có chuỗi đó |
| Chuỗi chưa cập nhật cùng ngày | Cập nhật lịch sử hoặc dự báo từng chuỗi bằng bộ lọc |
| `verify_run` báo CSV đã đổi | Dùng đúng CSV của lần train hoặc train lại với dữ liệu mới |
| Chưa tính được tồn kho | Xác nhận cùng đơn vị; kiểm tra số không âm và lead time ≤ horizon |
