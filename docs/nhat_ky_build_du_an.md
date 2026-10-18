# Nhật ký xây dựng dự án

**Dự án:** Hệ thống dự báo nhu cầu hàng hóa và hỗ trợ quản lý tồn kho bằng Machine Learning.
**Ngày tổng hợp:** 18/10/2026 (Asia/Ho_Chi_Minh).

Nhật ký tổng hợp các đầu việc và đầu ra phục vụ bảo trì, báo cáo và thuyết trình.
Timestamp từng lượt huấn luyện nằm trong `results/official/training_events.jsonl`;
nguồn dữ liệu, phạm vi, cấu hình và tổng thời gian nằm trong `run_manifest.json`.

## 1. Phạm vi thực nghiệm

| Thành phần | Dữ liệu / phạm vi | Đầu ra |
| --- | --- | --- |
| Dữ liệu gốc và EDA | Kaggle: 3.000.888 dòng, 54 cửa hàng × 33 nhóm, 01/01/2013–15/08/2017 | `data/raw/`, `results/official/metrics/eda*.json`, `results/official/figures/eda_*.html` |
| Huấn luyện chính thức | 54 cửa hàng × BEVERAGES, GROCERY I, PRODUCE; 162 chuỗi, 272.808 quan sát; không quick | [results/official](../results/official), `models/official/` |
| Huấn luyện nhanh khi vấn đáp (tùy chọn) | CSV Kaggle, cửa hàng 1 / BEVERAGES, quick | `results/live_training/`, `models/live_training/`; tách riêng kết quả chính thức |

Phạm vi ba nhóm được chọn trước khi xem metric. Chưa train đủ 33 nhóm hoặc đo tài nguyên
cho toàn bộ 1.782 chuỗi. EDA và mô hình vì vậy có phạm vi khác nhau.

## 2. Các công việc đã thực hiện

| Đầu việc | Cách thực hiện | Mã nguồn / bằng chứng |
| --- | --- | --- |
| Xác định bài toán | Dự báo sales theo cửa hàng × nhóm hàng × ngày; tiêu chí train, tuning, test và tồn kho | [Đối chiếu yêu cầu](doi_chieu_yeu_cau.md) |
| Thiết lập môi trường | Tạo `.venv`, cố định thư viện; thực nghiệm trên macOS arm64, Python 3.9.6 | [requirements.txt](../requirements.txt), [requirements-lock.txt](../requirements-lock.txt) |
| Kiểm tra dữ liệu gốc | Kiểm tra schema, ngày, sales, khóa trùng; ghi nguồn, đơn vị và SHA-256 | [src/data_processing.py](../src/data_processing.py), `data/raw/train.metadata.json` |
| Xử lý ngày thiếu | Bổ sung lịch ngày, forward fill làm state; ngày chèn không làm nhãn | `regularize_daily`, cờ `is_observed` |
| Tạo đặc trưng | Lag 1/7/14/28, rolling 7/14/28 với shift trước rolling, lịch và chu kỳ; thống nhất train/inference | [src/feature_engineering.py](../src/feature_engineering.py) |
| EDA toàn CSV | Phân tích ngày, cửa hàng, nhóm, thứ, tháng, khuyến mãi; ghép lễ đúng địa bàn | [src/eda.py](../src/eda.py), [eda.json](../results/official/metrics/eda.json) |
| Huấn luyện và tuning | Ridge, Random Forest, HGB; cấu hình gốc và 3 lượt random search mỗi thuật toán; seasonal naive làm mốc | [src/train.py](../src/train.py), [src/tuning.py](../src/tuning.py), [tuning.csv](../results/official/metrics/tuning.csv) |
| Chọn và đánh giá model | Chọn theo validation, khóa lựa chọn trước test; recursive holdout 30 ngày | [comparison.csv](../results/official/metrics/comparison.csv), [test_predictions.csv](../results/official/metrics/test_predictions.csv) |
| Ghi minh chứng huấn luyện | Timestamp từng lượt, số mẫu, tham số, thời gian, metric, snapshot mã và chẩn đoán HGB | [src/training_evidence.py](../src/training_evidence.py), [training.log](../results/official/training.log) |
| Lưu và kiểm chứng model | Tách model đánh giá với model triển khai; tái hiện holdout, kiểm tra dự báo 7/30 ngày | [src/verify_run.py](../src/verify_run.py), [verification.json](../results/official/metrics/verification.json) |
| CLI dự báo | Đọc lịch sử có sales, lọc chuỗi đã học, dự báo đệ quy và xuất CSV/metadata | [src/predict.py](../src/predict.py) |
| Tồn kho và giao diện | Sáu trang tiếng Việt; nhập tồn, lead time, hàng về, tồn an toàn; xác nhận đơn vị và tính Q | [app.py](../app.py), [src/inventory.py](../src/inventory.py) |
| Kiểm thử và tài liệu | Test dữ liệu, leakage, recursion, E2E, EDA, AppTest; báo cáo và hướng dẫn chạy | [tests](../tests), [kiểm thử](kiem_thu.md), [README](../README.md) |

## 3. Đầu ra chính thức

| Tập | Khoảng thời gian | Số quan sát thật trước loại warm-up |
| --- | --- | ---: |
| Train | 2013-01-01 → 2017-06-16 | 263.088 |
| Validation | 2017-06-17 → 2017-07-16 | 4.860 |
| Test | 2017-07-17 → 2017-08-15 | 4.860 |

Train sau tạo feature còn **258.552 mẫu**; 21 cột trước one-hot, 76 cột sau encoding.
Bốn ngày 25/12 của các năm 2013–2016 không có quan sát, tương ứng **648 ngày chèn** trên
162 chuỗi; chỉ dùng làm state. Loại **4.536 quan sát warm-up** không đủ lịch sử.

Random Forest được chọn theo validation. Bảng metric và phân tích nằm trong
[báo cáo chính](bao_cao_giua_ky.md), lấy từ `comparison.csv` và `tuning_summary.csv`.
Model đánh giá nạp lại tái hiện **4.860 dự báo holdout** trong dung sai 1e-8.
Model triển khai xuất **1.134 dòng cho 7 ngày** và **4.860 dòng cho 30 ngày**.
Dự báo bắt đầu 16/08/2017, sau ngày cuối dữ liệu.

Bộ kiểm thử đã ghi nhận **41 passed**; chi tiết và giới hạn xác minh ở
[nhật ký kiểm thử](kiem_thu.md). AppTest kiểm tra sáu trang và luồng dự báo/tồn kho;
chưa kiểm tra hình ảnh bằng trình duyệt thật.

## 4. Vấn đề và cách xử lý

| Vấn đề | Cách xử lý / kết quả |
| --- | --- |
| NumPy 2.0.2 phát RuntimeWarning phép nhân ma trận trên máy thử | Chuyển sang 1.26.4, train và kiểm thử lại; không còn cảnh báo số học này |
| Lag có thể đếm theo dòng thay vì ngày | Regularize lịch ngày và đánh dấu ngày chèn; lag đúng ngày, nhãn chèn bị loại |
| CSV lớn làm biểu đồ quá nặng | Tính quartile/whisker trên toàn CSV thay vì đưa hàng triệu điểm vào HTML |
| Test UI chọn nhầm selectbox | Chọn widget theo nhãn “Số ngày dự báo”; kiểm tra đổi 7/30 ngày và xóa kết quả cũ |
| Model refit toàn lịch sử không dùng để tái hiện test | Giữ `evaluation_model.joblib` riêng và kiểm chứng bằng `verify_run` |
| Huấn luyện dài khi trình bày | Chuẩn bị lệnh quick một chuỗi Kaggle, kết quả lưu riêng; xem [minh chứng train](minh_chung_huan_luyen.md) |
| joblib không đọc được số lõi vật lý trong sandbox macOS | Ghi nhận warning, giữ RF và native pool giới hạn 1; test vẫn qua |

## 5. Chạy lại và công việc tiếp theo

Lệnh cài đặt và chạy toàn bộ quy trình nằm trong
[README, mục 3](../README.md#3-chạy-toàn-bộ-chương-trình-từ-đầu).
Đổi thư mục output/model nếu cần giữ nhiều lần thực nghiệm; sinh lại báo cáo sau khi train hoàn tất.

Công việc còn lại: mở rộng đủ 33 nhóm, backtest nhiều origin, thêm khoảng dự báo và biến ngoại sinh;
kiểm tra nền tảng khác và bố cục trình duyệt; bổ sung dữ liệu tồn theo SKU/chi phí nếu tối ưu tồn kho.
Tên và đóng góp thực tế của thành viên cần xác nhận trong [bảng phân công](phan_cong_nhiem_vu.md).

Khi cập nhật nhật ký, ghi ngày, mục tiêu, thay đổi, output, cách kiểm chứng và giới hạn còn lại.
File này cập nhật thủ công; `src.report` chỉ sinh báo cáo giữa kỳ.
