# Nhật ký xây dựng dự án

**Dự án:** Hệ thống dự báo nhu cầu hàng hóa hỗ trợ quản lý tồn kho bằng Machine Learning.  
**Ngày tổng hợp:** 18/10/2026. **Múi giờ:** Asia/Ho_Chi_Minh (UTC+7).  
**Trạng thái:** đã triển khai pipeline, thực nghiệm trên dữ liệu Kaggle, giao diện và kiểm thử.

Nhật ký này tổng hợp công việc đã thực hiện, mô tả cách làm và đầu ra để tra cứu khi bảo trì,
viết báo cáo hoặc thuyết trình. Các bước phát triển được ghi theo thứ tự triển khai;
mốc giờ chính xác của lần train Kaggle lấy từ `training_events.jsonl` và chuyển sang UTC+7.
Không gán giờ thực hiện cho các công việc không có log thời gian.

## 1. Phân biệt các lần chạy

| Lần chạy | Dữ liệu và phạm vi | Nơi lưu đầu ra | Mục đích |
| --- | --- | --- | --- |
| Demo mô phỏng | 4.320 dòng; 3 cửa hàng × 3 nhóm; 480 ngày | [results/demo](../results/demo), [models/demo](../models/demo) | Hoàn thiện và kiểm tra pipeline khi chưa có CSV thật |
| Kaggle chính thức | 272.808 quan sát; 54 cửa hàng × 3 nhóm; không quick | [results/official](../results/official), [models/official](../models/official) | Thực nghiệm chính để đưa vào báo cáo |
| Demo train trực tiếp | CSV Kaggle; cửa hàng 1, BEVERAGES; quick | [results/live_training](../results/live_training), [models/live_training](../models/live_training) | Trình diễn fit/tuning nhanh khi vấn đáp |

CSV Kaggle đầy đủ có **3.000.888 dòng, 54 cửa hàng, 33 nhóm hàng**. EDA dùng toàn bộ CSV;
mô hình chính thức chỉ học ba nhóm **BEVERAGES, GROCERY I, PRODUCE** ở cả 54 cửa hàng.
Số liệu của ba lần chạy được lưu riêng và không thay thế cho nhau trong báo cáo.

## 2. Giai đoạn A — Xây dựng nền tảng và bản demo

| Mã | Đầu việc | Mô tả công việc đã thực hiện | Output / kết quả |
| --- | --- | --- | --- |
| A01 | Đọc yêu cầu và xác định bài toán | Đối chiếu tài liệu giữa kỳ, đặc tả và yêu cầu người dùng; xác định dự báo theo cửa hàng × nhóm hàng × ngày; không dùng API LLM cho bài toán chính | Phạm vi, tiêu chí và bằng chứng trong [đối chiếu yêu cầu](doi_chieu_yeu_cau.md) |
| A02 | Kiểm tra và thiết lập môi trường | Kiểm tra Python/thư viện; tạo `.venv`; cài thư viện xử lý dữ liệu, ML, biểu đồ, Streamlit và pytest | [requirements.txt](../requirements.txt), [requirements-lock.txt](../requirements-lock.txt); môi trường đã chạy: Python 3.9.6, macOS arm64 |
| A03 | Tổ chức thư mục | Tách dữ liệu gốc/demo/trung gian, module xử lý/train/predict, model, kết quả, tests và tài liệu | `data/`, `src/`, `models/`, `results/`, `tests/`, `docs/`, `app.py` |
| A04 | Tạo bộ dữ liệu mô phỏng | Sinh dữ liệu bán hàng có xu hướng, chu kỳ và khuyến mãi, seed 42; ghi rõ nguồn và đơn vị mô phỏng | [src/demo_data.py](../src/demo_data.py), [sales.csv](../data/demo/sales.csv), [metadata demo](../data/demo/sales.metadata.json); 4.320 dòng |
| A05 | Viết xử lý và kiểm tra CSV | Kiểm tra cột, ngày, khóa trùng, sales thiếu/âm/vô hạn, mã cửa hàng và nhóm hàng; sắp theo thời gian | [src/data_processing.py](../src/data_processing.py); đầu vào sai bị từ chối với thông báo cụ thể |
| A06 | Xử lý ngày thiếu | Bổ sung lịch ngày cho từng chuỗi, forward fill chỉ để duy trì trạng thái quá khứ; đánh dấu ngày chèn không phải quan sát thật | `regularize_daily`, cờ `is_observed`; ngày chèn không được dùng làm nhãn train/test |
| A07 | Tạo đặc trưng | Tạo lịch, chu kỳ sin/cos, trend, lag 1/7/14/28 và rolling mean/std 7/14/28; shift trước rolling; dùng cùng công thức ở train và inference | [src/feature_engineering.py](../src/feature_engineering.py); có kiểm tra không leakage và feature parity |
| A08 | Xây dựng các mô hình | Đóng gói scaler, one-hot và estimator thành sklearn Pipeline; xây Ridge, Random Forest, HGB; thêm seasonal naive lấy lag_7 | [src/models.py](../src/models.py); ba thuật toán ML và một mốc đối chiếu không cần fit |
| A09 | Viết chương trình train và tuning | Chia theo ngày; fit từng cấu hình trên train; tìm tham số bằng ParameterSampler và recursive validation; khóa lựa chọn trước test | [src/train.py](../src/train.py), [src/tuning.py](../src/tuning.py); seed 42, lưu kết quả từng lượt |
| A10 | Viết đánh giá và lưu model | Tính MAE, RMSE, WAPE; đo tốc độ; lưu dự báo và biểu đồ; tách model đánh giá với model triển khai refit sau test | [src/evaluate.py](../src/evaluate.py), `comparison.csv`, `test_predictions.csv`, hai file `.joblib` theo lần chạy |
| A11 | Viết pipeline suy luận | Đọc CSV lịch sử mới, kiểm tra chuỗi đã học, lấy 28 ngày gần nhất; dự báo đệ quy và cập nhật lag bằng dự báo đã sinh; xuất CSV và metadata | [src/predict.py](../src/predict.py); CLI nhận input → tiền xử lý → model → dự báo |
| A12 | Xây chức năng tồn kho | Nhập tồn hiện tại, lead time, tồn an toàn, hàng đang về; xác nhận cùng đơn vị; tính Q = max(0, D + S − C − I); từ chối lead time vượt horizon | [src/inventory.py](../src/inventory.py); ví dụ SRS trả Q = 120; có cảnh báo thiếu/dư theo quy tắc |
| A13 | Xây giao diện | Tạo sáu trang tiếng Việt; lọc dữ liệu, dự báo, biểu đồ, tải CSV, nhập tồn kho, xem metric và cấu hình | [app.py](../app.py), [.streamlit/config.toml](../.streamlit/config.toml); giao diện chỉ nạp model đã train |
| A14 | Viết tài liệu và kiểm thử ban đầu | Viết README, báo cáo, phân công, thuyết trình; chạy test lõi, E2E và AppTest; xử lý các lỗi phát hiện | [tests](../tests), [README](../README.md), [báo cáo demo được lưu riêng](bao_cao_demo.md); bản demo hoàn chỉnh trước khi nhận CSV Kaggle |

## 3. Giai đoạn B — Chuyển sang dữ liệu Kaggle thực

| Mã | Đầu việc | Mô tả công việc đã thực hiện | Output / kết quả |
| --- | --- | --- | --- |
| B01 | Kiểm tra dữ liệu người dùng tải | Đọc schema và kiểm tra toàn bộ train.csv; kiểm tra stores.csv và holidays_events.csv; giữ nguyên CSV gốc | 3.000.888 dòng, 1.684 ngày quan sát, 1.782 chuỗi; không missing value hoặc trùng khóa; ngày 2013-01-01–2017-08-15 |
| B02 | Ghi nhận nguồn dữ liệu | Ghi nguồn Kaggle theo người dùng xác nhận, đơn vị sales tổng hợp, SHA-256 CSV và phạm vi mô hình | [data/raw/train.metadata.json](../data/raw/train.metadata.json); hash được đưa vào manifest để kiểm tra tái lập |
| B03 | Chạy EDA toàn bộ CSV | Phân tích ngày, cửa hàng, nhóm hàng, thứ, tháng, khuyến mãi; ghép lễ đúng địa bàn, loại transferred và Work Day | [eda.json](../results/official/metrics/eda.json), các `eda_*.csv` trong metrics và 7 biểu đồ `eda_*.html` trong [figures](../results/official/figures) |
| B04 | Điều chỉnh biểu đồ cho CSV lớn | Tính quartile và whisker trên toàn dữ liệu thay vì đưa hàng triệu điểm vào HTML; không lấy mẫu để thay cho thống kê phân phối | [src/eda.py](../src/eda.py), [eda_distributions.json](../results/official/metrics/eda_distributions.json); HTML gọn hơn, thống kê vẫn dùng toàn bộ dữ liệu |
| B05 | Xác định phạm vi train | Chọn cả 54 cửa hàng và ba nhóm trước khi xem metric; chia thời gian, tạo feature, loại warm-up và nhãn ngày chèn | 162 chuỗi; 272.808 quan sát; 258.552 mẫu train sau feature engineering; chi tiết ở [data_quality.json](../results/official/metrics/data_quality.json) |
| B06 | Bổ sung bằng chứng train | Ghi sự kiện từng lượt fit, timestamp, số mẫu, tham số đầy đủ, thời gian và metric; chụp lại mã nguồn; chẩn đoán HGB trên tối đa 5.000 mẫu train cố định | [src/training_evidence.py](../src/training_evidence.py), [training.log](../results/official/training.log), [training_events.jsonl](../results/official/training_events.jsonl), [source_snapshot](../results/official/source_snapshot), [training_history.csv](../results/official/metrics/training_history.csv) |
| B07 | Train và tuning chính thức | Fit ba thuật toán, mỗi thuật toán có cấu hình gốc và 3 lượt random search; chấm cùng validation 30 ngày; naive chấm riêng | 12 lượt fit cấu hình ML và 1 cấu hình naive; [tuning.json](../results/official/metrics/tuning.json), [tuning.csv](../results/official/metrics/tuning.csv), [tuning_summary.csv](../results/official/metrics/tuning_summary.csv) |
| B08 | Lựa chọn mô hình | So WAPE validation; dùng MAE nếu WAPE không xác định; chỉ sau khi khóa lựa chọn mới chạy test | Chọn Random Forest, WAPE validation 12,6005%; cấu hình cụ thể ở mục 5 dưới đây |
| B09 | Đánh giá test giữ riêng | Refit cấu hình tốt nhất của từng thuật toán trên train + validation; dự báo đệ quy cùng test 30 ngày | 4.860 quan sát test; [comparison.csv](../results/official/metrics/comparison.csv), [test_predictions.csv](../results/official/metrics/test_predictions.csv), [test_by_series.csv](../results/official/metrics/test_by_series.csv), [biểu đồ thực tế–dự báo](../results/official/figures/test_actual_vs_forecast.html) |
| B10 | Lưu hai artefact | Giữ model fit trước test để tái hiện đánh giá; sau đánh giá refit model triển khai trên toàn bộ lịch sử thuộc phạm vi đã chọn | [evaluation_model.joblib](../models/official/evaluation_model.joblib) và [model.joblib](../models/official/model.joblib); không dùng model refit toàn lịch sử để báo lại test |
| B11 | Kiểm chứng bằng CSV thật | So hash đầu vào, nạp lại model đánh giá và tái hiện holdout; nạp model triển khai để sinh 7/30 ngày cho mọi chuỗi | [src/verify_run.py](../src/verify_run.py), [verification.json](../results/official/metrics/verification.json); tái hiện 4.860 dự báo test; xuất 1.134 dòng cho 7 ngày và 4.860 dòng cho 30 ngày |
| B12 | Kiểm tra CLI và giao diện chính thức | Chạy CLI lọc cửa hàng 1 / BEVERAGES; chạy AppTest sáu trang trên CSV thật, bấm dự báo 30 ngày và gửi form tồn kho | [forecast_30_days.csv](../results/official/forecast_30_days.csv), metadata đi kèm và [app_verification.json](../results/official/metrics/app_verification.json); không có exception/error trong các luồng đã kiểm tra |
| B13 | Chuẩn bị demo train tại lớp | Chạy quick trên một chuỗi Kaggle; dùng thư mục riêng để không thay kết quả chính thức | [manifest live demo](../results/live_training/run_manifest.json), [log live demo](../results/live_training/training.log); thời gian đo 4,906 giây trên máy thử nghiệm |
| B14 | Cập nhật hồ sơ bài nộp | Sinh lại báo cáo từ số đo Kaggle; thêm minh chứng train; cập nhật README, đối chiếu yêu cầu, thuyết trình và kiểm thử | [báo cáo chính](bao_cao_giua_ky.md), [minh chứng train](minh_chung_huan_luyen.md), [đối chiếu](doi_chieu_yeu_cau.md), [thuyết trình](huong_dan_thuyet_trinh.md), [kiểm thử](kiem_thu.md); tên thành viên/phân công thực tế để nhóm xác nhận |
| B15 | Tổng hợp nhật ký build | Ghi lại đầu việc, cách thực hiện, đầu ra, mốc train và giới hạn; bổ sung liên kết từ README | File nhật ký này; duy trì riêng với log huấn luyện và nhật ký kiểm thử |

## 4. Mốc thực thi của lần train chính thức

Nguồn: [training_events.jsonl](../results/official/training_events.jsonl). Các giờ sau cùng ngày **18/10/2026**, UTC+7.

| Giờ | Sự kiện | Đầu ra / ý nghĩa |
| --- | --- | --- |
| 11:15:40 | Bắt đầu lần chạy | Ghi input, bộ lọc, horizon 30 ngày, 3 lượt tuning, quick=False |
| 11:15:42 | Hoàn tất chuẩn bị dữ liệu | Chốt 272.808 quan sát, 162 chuỗi và mốc train/validation/test |
| 11:15:43 | Hoàn tất tạo đặc trưng train | 258.552 mẫu; 21 cột trước one-hot |
| 11:31:24 | Khóa lựa chọn mô hình | Random Forest được chọn theo validation, trước khi đánh giá test |
| 11:31:24–11:34:08 | Đánh giá test các thuật toán | Lần lượt seasonal naive, Ridge, Random Forest và HGB |
| 11:34:08 | Bắt đầu refit để triển khai | Random Forest học từ 268.272 quan sát có đủ feature trong toàn lịch sử đã chọn |
| 11:36:39 | Lưu/nạp model và kết thúc | Dự báo trước/sau nạp khớp; ghi manifest, bảng metric, đồ thị và artefact |

Thời gian chương trình đo bằng `perf_counter`: **1.151,051 giây**, khoảng **19,18 phút**.
Đây là số đo trong manifest, không lấy hiệu các timestamp UTC để thay thế số đo này.

## 5. Đầu ra thực nghiệm chính

### Chia dữ liệu và tạo đặc trưng

| Tập | Khoảng thời gian | Số quan sát thật trước loại warm-up |
| --- | --- | ---: |
| Train | 2013-01-01 → 2017-06-16 | 263.088 |
| Validation | 2017-06-17 → 2017-07-16 | 4.860 |
| Test | 2017-07-17 → 2017-08-15 | 4.860 |

Train sau tạo feature còn **258.552 mẫu**. Có **21 cột trước one-hot**, gồm 19 biến số và 2 khóa phân loại;
với 54 cửa hàng và 3 nhóm, model nhận 76 cột sau encoding.
Bốn ngày 25/12 của các năm 2013–2016 không có quan sát, tương ứng **648 ngày chèn** trên 162 chuỗi.
Các ngày này chỉ làm trạng thái lịch sử; không được coi là nhãn thật.
Loại **4.536 quan sát warm-up** không đủ 28 ngày lịch sử.

### Tuning và kết quả test

| Thuật toán | WAPE validation trước tuning (%) | WAPE validation tốt nhất (%) | MAE test | RMSE test | WAPE test (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Seasonal naive | 14,5314 | 14,5314 | 517,5552 | 899,2306 | 14,9862 |
| Ridge | 15,9947 | 15,6898 | 542,3826 | 797,5374 | 15,7051 |
| Random Forest | 12,6153 | 12,6005 | 445,6334 | 692,4956 | 12,9037 |
| HistGradientBoosting | 14,4865 | 13,4358 | 476,4709 | 732,5128 | 13,7966 |

Seasonal naive không có bước fit/tuning. Random Forest được chọn bằng validation, với
`n_estimators=160`, `max_depth=10`, `min_samples_leaf=6`, `max_features=0.7`, `random_state=42`, `n_jobs=1`.
Lần refit train + validation để đánh giá test mất **148,375 giây**;
suy luận toàn bộ test 30 ngày mất **0,439 giây**, theo phạm vi đo được giải thích trong README.
Các số liệu ở bảng lấy từ comparison.csv và tuning_summary.csv, không phải kết quả mô phỏng.

### Kiểm chứng và dự báo mới

- Bộ pytest cuối đã chạy: **41 passed, 0 failures, 0 errors, 0 skipped**;
  xem [JUnit XML](../results/metrics/pytest.xml) và [nhật ký kiểm thử](kiem_thu.md).
- Model đánh giá nạp lại tái hiện 4.860 dự báo holdout trong dung sai kiểm tra 1e-8.
- Dự báo 7 ngày: **1.134 dòng**, 2017-08-16 → 2017-08-22;
  [CSV](../results/official/forecast_7_days_all_series.csv).
- Dự báo 30 ngày: **4.860 dòng**, 2017-08-16 → 2017-09-14;
  [CSV](../results/official/forecast_30_days_all_series.csv).
- Streamlit AppTest trên dữ liệu thật đã mở sáu trang và kiểm tra luồng dự báo/tồn kho.
  HTTP 200 của server cục bộ được ghi nhận trong giai đoạn xây dựng trước; chưa kiểm tra hình ảnh bằng trình duyệt thật.

## 6. Vấn đề phát hiện và cách xử lý

| Vấn đề | Cách xử lý | Kết quả ghi nhận |
| --- | --- | --- |
| Ban đầu chưa có CSV Kaggle | Hoàn thiện code bằng demo có metadata mô phỏng; chuyển sang dữ liệu thật sau khi người dùng cung cấp | Tách demo/official/live_training; giữ báo cáo demo riêng |
| NumPy 2.0.2 phát RuntimeWarning phép nhân ma trận trên máy thử | Chuyển sang NumPy 1.26.4, train lại và kiểm thử lại | Lần thực nghiệm cuối không còn cảnh báo số học này |
| Test UI chọn nhầm selectbox khi đổi horizon | Sửa test chọn widget theo nhãn “Số ngày dự báo” | Luồng đổi 7/30 ngày và việc bỏ kết quả cũ đã được kiểm tra |
| Ngày thiếu có thể làm lag_7 thành 7 dòng thay vì 7 ngày | Regularize lịch ngày, forward fill quá khứ và đánh dấu ngày chèn | Lag đúng theo ngày lịch, nhãn ngày chèn bị loại khỏi train/test |
| CSV lớn khiến box plot chứa quá nhiều điểm | Xuất quartile/whisker tính trên toàn bộ dữ liệu | Giữ thống kê đầy đủ, giảm kích thước dữ liệu gửi vào HTML |
| Cấu hình RF sâu hơn tốn thời gian nhưng không tốt hơn | Ghi cả cấu hình chậm/kém và chọn bằng validation | Depth 16 mất khoảng 301 giây, WAPE 12,6603%; cấu hình depth 10 đạt 12,6005% |
| Model refit toàn lịch sử không thích hợp để tái hiện test | Lưu riêng evaluation_model và model triển khai | verify_run tái hiện holdout bằng đúng evaluation_model |
| Train chính thức quá dài để trình diễn toàn bộ tại lớp | Chuẩn bị lệnh quick một chuỗi, đầu ra lưu riêng | Đã chạy thành công trong khoảng 4,91 giây |
| joblib không đọc được số lõi vật lý trong sandbox macOS | Ghi nhận warning; giữ RF và native pool giới hạn 1 | Test vẫn qua; chi tiết trong kiem_thu.md |

## 7. Các lệnh đã sử dụng để tái lập

Chạy tại thư mục gốc, sau khi kích hoạt `.venv`. Các lệnh train ghi đè kết quả cùng thư mục;
đổi `--output` và `--model-dir` nếu muốn giữ nhiều lần thực nghiệm.

```bash
python -m src.eda --data data/raw/train.csv --output results/official --holidays data/raw/holidays_events.csv --stores data/raw/stores.csv
python -m src.train --data data/raw/train.csv --output results/official --model-dir models/official --horizon 30 --trials 3 --families BEVERAGES 'GROCERY I' PRODUCE
python -m src.evaluate --predictions results/official/metrics/test_predictions.csv --output results/official/metrics/recomputed_test.json
python -m src.verify_run --results results/official
python -m src.predict --input data/raw/train.csv --model models/official/model.joblib --store 1 --family BEVERAGES --horizon 30 --output results/official/forecast_30_days.csv
python -m src.report --results results/official --output docs/bao_cao_giua_ky.md
python -m pytest -q --junitxml=results/metrics/pytest.xml
```

Demo train trực tiếp tại lớp:

```bash
python -m src.train --data data/raw/train.csv --output results/live_training --model-dir models/live_training --stores 1 --families BEVERAGES --quick --trials 1 --horizon 30
```

Mở giao diện bằng `python -m streamlit run app.py`.

## 8. Giới hạn và công việc còn lại

| Nội dung | Trạng thái hiện tại / đầu ra cần có khi thực hiện tiếp |
| --- | --- |
| Mở rộng phạm vi model | Chưa train đủ 33 nhóm; cần lần chạy và bảng metric mới cho phạm vi mở rộng |
| Backtest nhiều thời điểm | Hiện một origin validation và một test; cần rolling-origin và đánh giá theo horizon |
| Khoảng dự báo / biến ngoại sinh | Chưa có khoảng dự báo; chưa dùng kế hoạch khuyến mãi/ngày lễ làm feature tương lai |
| Kiểm tra nền tảng và hình ảnh UI | Đã chạy macOS Python 3.9.6 và AppTest; chưa xác minh Windows/Linux/Python 3.11 hoặc bố cục trong trình duyệt thật |
| Tồn kho thực tế và tối ưu chi phí | Hiện người dùng nhập tồn kho và xác nhận đơn vị; chưa có dữ liệu tồn theo SKU, chi phí hoặc hàm mục tiêu tối ưu |
| Hoàn thiện thông tin nhóm | [Bảng phân công](phan_cong_nhiem_vu.md) còn tên và mức đóng góp để thành viên xác nhận |

Dữ liệu kết thúc 15/08/2017 nên các dự báo trên bắt đầu 16/08/2017; muốn dự báo cho hiện tại cần
lịch sử mới và đánh giá/huấn luyện lại phù hợp.

## 9. Cách cập nhật nhật ký sau này

Mỗi đầu việc mới cần ghi **ngày thực hiện, mã công việc, mục tiêu, mô tả thay đổi, output cụ thể,
cách kiểm chứng, kết quả và giới hạn còn lại**. Dẫn tới file/log liên quan; nếu thay model cần ghi cả
dataset, phạm vi, cấu hình và thư mục kết quả của lần chạy mới. Chỉ ghi hoàn tất sau khi có đầu ra.

File này là nhật ký tổng hợp được cập nhật thủ công. `src.report` sinh báo cáo giữa kỳ và không ghi đè
nhật ký này. Log từng lần fit nằm ở training.log/JSONL; chi tiết test nằm ở kiem_thu.md và JUnit XML.
