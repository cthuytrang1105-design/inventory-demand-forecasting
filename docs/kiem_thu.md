# Nhật ký kiểm thử thực tế

Ngày thực thi: **18/10/2026** (múi giờ người dùng Asia/Ho_Chi_Minh).
Môi trường: macOS arm64, Python 3.9.6; NumPy 1.26.4, pandas 2.2.3,
scikit-learn 1.6.1, Streamlit 1.49.1. Xem requirements-lock.txt và results/official/run_manifest.json.

## Kết quả xác nhận

| Kiểm tra đã chạy | Kết quả thực tế |
| --- | --- |
| `python -m pytest -q --junitxml=results/metrics/pytest.xml` | **41 passed**, 1 warning, 6,04 giây |
| `python -m pip check` | No broken requirements found |
| Kiểm tra AST mọi app/src/tests Python | Tất cả file phân tích cú pháp thành công |
| `src.data_processing` trên data/raw/train.csv | CSV Kaggle 3.000.888 dòng không missing/trùng khóa, đã kiểm tra trước train |
| `src.train` trên CSV Kaggle, không quick | 3 thuật toán ML + naive; 9 lượt random search, 3 cấu hình ML gốc; 54 cửa hàng × 3 nhóm, validation/test mỗi tập 4.860 quan sát |
| Lưu/nạp model trong train | Dự báo trước/sau joblib khớp; đã tạo hai artefact evaluation/deployment |
| `src.evaluate` từ test_predictions.csv | Kaggle: MAE 445,6333614054; RMSE 692,4955666579; WAPE 12,9036736764% |
| `src.predict` bằng CSV từ đĩa, cửa hàng 1 / BEVERAGES / 30 ngày | 30 dòng CSV và metadata nguồn được xuất thành công |
| `src.eda` | 7 biểu đồ HTML trên CSV Kaggle, gồm lễ theo địa bàn; JSON thống kê và CSV tổng hợp; box plot dùng quantile toàn bộ dữ liệu |
| `src.report` | Báo cáo 16 mục sinh từ manifest/metric thực tế |
| Khởi động server Streamlit cục bộ | Server khởi động tại 127.0.0.1:8501 |
| HTTP `/_stcore/health` và `/` | Cả hai trả **HTTP 200**, health body `ok` |

HTTP được xác nhận sau khi cấp quyền mở/kiểm tra server bên ngoài sandbox.
AppTest chạy nội bộ thực thi mã ứng dụng; chưa kiểm tra giao diện bằng trình duyệt thật hoặc chụp ảnh.
Server kiểm tra được dừng sau khi xác minh; người dùng khởi động lại bằng lệnh trong README.

## Phạm vi 41 test tự động

- Schema, đọc CSV mới, sắp thời gian; từ chối sales âm/NaN/vô hạn, ngày sai, family rỗng,
  mã cửa hàng không nguyên, khuyến mãi âm và khóa trùng.
- Thiếu cột khuyến mãi/missing khuyến mãi được xử lý; thiếu sales bị từ chối.
- Sửa nhãn tại ngày t và tương lai không đổi đặc trưng tại/đến t; nhóm khác độc lập.
- Ngày thiếu được chèn đúng lịch, không làm nhãn; lag_7 đúng khoảng cách 7 ngày.
- Feature train và inference khớp số học; mốc train/validation/test không chồng lấn.
- Recursive 30 ngày dùng chính dự báo trước đó, seasonal naive lặp lại chu kỳ 7 ngày.
- Từ chối horizon không hợp lệ, chuỗi chưa train, chuỗi quá ngắn/cũ và model không tồn tại.
- Công thức tồn kho ví dụ SRS trả đúng Q=120; cảnh báo dư; từ chối đầu vào âm,
  thiếu xác nhận đơn vị, lead time 0/vượt horizon và số không hữu hạn.
- Metric WAPE với mẫu số 0 trả None; MAE/RMSE/WAPE khớp ví dụ tính tay.
- Pipeline E2E: CSV → train cả ba thuật toán → tuning → test → lưu/nạp → CSV mới → CLI forecast.
  Tái hiện chính xác dự báo holdout bằng evaluation_model (không dùng model đã refit toàn lịch sử).
- EDA ngày lễ chỉ gắn Local/Regional đúng địa bàn, loại transferred/Work Day;
  xuất đủ 6 biểu đồ khi không có holidays CSV.
- AppTest mở đủ sáu trang, bấm dự báo 7 và 30 ngày, không hiển thị kết quả cũ sau đổi horizon,
  gửi biểu mẫu tồn kho khi chưa/vừa xác nhận đơn vị, hiển thị lỗi khi thiếu dữ liệu chính thức.

## Các vấn đề đã xử lý

Lần chạy đầu trên NumPy 2.0.2 có RuntimeWarning phép nhân ma trận dù đầu ra hữu hạn.
Đã chuyển sang NumPy 1.26.4, train lại và chạy lại test; cảnh báo số học không còn.
Một test UI ban đầu chọn nhầm selectbox nguồn khi đổi horizon; đã sửa test chọn theo nhãn widget.
Phiên bản cập nhật 41 test đều qua; JUnit XML là kết quả lần chạy cuối, không dùng kết quả lần lỗi.

Còn một UserWarning của joblib vì sandbox macOS không đọc được số lõi vật lý,
joblib dùng số lõi logic thay thế. Không có test fail. RF n_jobs và native pool trong train
đã giới hạn 1; không tắt cảnh báo để che giấu lỗi.

## Những phần chưa xác minh

Đã có CSV Kaggle và đã EDA toàn bộ dữ liệu/ngày lễ thực tế. Phạm vi train chính thức là 54 cửa hàng × 3 nhóm;
chưa train đủ 33 nhóm hoặc đo tài nguyên cho toàn bộ 1.782 chuỗi. Chưa backtest nhiều origin.
Chưa kiểm thử trên Windows/Linux/Python 3.11, chưa chạy trình duyệt thật để đánh giá bố cục.
Không có kiểm chứng tối ưu chi phí tồn kho hoặc khả năng tránh thiếu hàng trong vận hành thực tế.

Để tái lập: làm theo README, xem `results/metrics/pytest.xml` và
`results/official/metrics/comparison.csv`, `tuning.json`, `tuning_summary.csv`, `test_predictions.csv`.

## Kiểm chứng bổ sung sau khi hoàn tất thực nghiệm Kaggle

Bộ pytest cuối: **41 passed**, 0 failures, 0 errors, 0 skipped; JUnit 6.035 giây. Một warning joblib như đã mô tả.

Lần train chính thức không quick, 160 cây RF / 180 vòng HGB, 3 lượt random search mỗi thuật toán. Chọn **RandomForest** theo validation; tổng thời gian từng lần chạy ghi trong `run_manifest.json`.

Các kiểm tra sau đã thực sự chạy:

- `python -m src.verify_run --results results/official`: hash CSV khớp; evaluation_model nạp lại tái hiện toàn bộ 4.860 dự báo holdout trong dung sai 1e-8.
- Model triển khai sinh 1.134 dòng cho 7 ngày và 4.860 dòng cho 30 ngày, không trùng khóa và không có dự báo vô hạn; CSV ở results/official/forecast_*_days_all_series.csv.
- `src.predict` đọc lại train.csv, lọc cửa hàng 1 / BEVERAGES, xuất đúng 30 dòng và sidecar metadata.
- Streamlit AppTest trên CSV thật 3.000.888 dòng: nguồn mặc định Kaggle; mở đủ 6 trang không exception/error; nhấn dự báo 30 ngày trả 30 dòng; gửi biểu mẫu tồn kho hợp lệ trả 3 metric. Lưu tại results/official/metrics/app_verification.json.
- Chưa kiểm tra hình ảnh bằng trình duyệt thật. HTTP 200 ghi ở bảng trên là kiểm tra server cục bộ ở lần xây dựng trước; kiểm chứng dữ liệu Kaggle lần này dùng AppTest.

Lệnh huấn luyện nhanh một chuỗi Kaggle đã chạy thành công trong khoảng **4,91 giây**; log/artefact lưu riêng ở `results/live_training` và `models/live_training`. Đây là kiểm tra phạm vi nhỏ; bảng metric báo cáo dùng lần chạy chính thức 162 chuỗi.
