# Thể hiện rõ quá trình huấn luyện trong bài nộp

Yêu cầu của giảng viên cần được thể hiện bằng **mã học từ dữ liệu, thiết kế thực nghiệm và kết quả
chạy có thể đối chiếu**. Một website dự báo hay file model.joblib đơn lẻ chưa giải thích được
model đã được tạo ra như thế nào. Không cần mọi thuật toán đều có epoch như mạng nơ-ron.

## 1. Nội dung cần có trong báo cáo

| Thành phần | Nội dung trình bày cho dự án này | Vị trí bằng chứng |
| --- | --- | --- |
| Dữ liệu train | CSV Kaggle 3.000.888 dòng; phạm vi model 54 cửa hàng × 3 nhóm; chọn trước khi chấm điểm | Mục 5–6 báo cáo, run_manifest.json |
| Tạo dữ liệu học | X gồm lịch, lag, rolling, cửa hàng/nhóm; y là sales quan sát thật; loại warm-up/ngày chèn | Mục 7–8, features_ready trong JSONL |
| Chia dữ liệu | Train 2013-01-01–2017-06-16; validation 2017-06-17–2017-07-16; test 2017-07-17–2017-08-15 | Bảng ở mục 8, temporal_split.html |
| Thực sự fit model | Chỉ rõ `.fit(X_train, y_train)` trong src/tuning.py và Pipeline của src/models.py | Mục 8.1; source_snapshot/src/ |
| Tuning | Tham số khảo sát, seed 42, mỗi cấu hình fit riêng, recursive validation, bảng trước/sau | Mục 9, tuning.json/csv, tuning_summary.csv |
| Kết quả học và chọn model | Loss train chỉ dùng chẩn đoán; chọn bằng WAPE validation, không dùng test | Mục 8.2–10, training_history.csv |
| Đánh giá cuối | MAE/RMSE/WAPE test, thời gian train/suy luận; phân tích naive so với ML | Mục 10, comparison.csv |
| Artefact tái lập | Model đã lưu/nạp lại, lệnh chạy, môi trường, SHA-256, kiểm tra khớp dự báo holdout | run_manifest.json, verification.json |

Số mẫu train sau tạo đặc trưng là **258.552**, validation **4.860**, test **4.860**.
Có **21 cột đầu vào trước one-hot**, tương ứng 19 biến số và 2 khóa phân loại;
one-hot 54 cửa hàng + 3 nhóm tạo 57 biến, tổng cộng 76 cột model ở phạm vi này.
Không lấy số liệu từ lần demo mô phỏng để điền vào bảng thực nghiệm Kaggle.

## 2. Cách giải thích đoạn mã train khi vấn đáp

Mở `src/tuning.py`, hàm `search_model`, tại đoạn:

```python
model = build_model(name, params, quick)
model.fit(train_features[FEATURES], train_features.sales)
pred = recursive_forecast(model, history, horizon, name)
matched = join_actual(validation, pred)
```

Giải thích: `train_features[FEATURES]` là X_train và `train_features.sales` là y_train.
`.fit` học cả preprocessing có trạng thái và estimator. Sau fit, chương trình dự báo từ cuối
train qua toàn bộ 30 ngày validation. Sales thật trong validation chỉ dùng chấm metric,
không được cấp lại làm lag cho các bước sau. Mỗi cấu hình tham số tạo một model mới.

- **Ridge:** học hệ số hồi quy và intercept với phạt L2; không cần vòng lặp epoch hiển thị.
- **Random Forest:** học các cây và luật chia nhánh từ tập train, số cây/độ sâu/min leaf được công bố.
- **HGB:** xây cây theo các vòng boosting; có thể trình bày RMSE train theo iteration.
- **Seasonal naive:** không fit. Dùng lag_7 làm mốc; không tính naive thành thuật toán ML đã huấn luyện.

## 3. Hình nên chèn khi chuyển báo cáo sang Word/PDF

Các hình tương tác nằm trong `results/official/figures/`:

1. **temporal_split.html:** cho thấy thứ tự train → validation → test, ghi rõ ngày và số mẫu.
2. **training_convergence.html:** quá trình HGB thay đổi RMSE trên tối đa 5.000 mẫu train cố định.
   Chú thích: “Chỉ số in-sample phục vụ chẩn đoán; không dùng thay metric trên dữ liệu chưa thấy”.
3. **tuning_validation.html:** WAPE validation từng cấu hình của ba mô hình và naive.
   Mỗi điểm là một lần fit riêng, không gọi các cấu hình là các epoch của cùng một model.
4. **test_actual_vs_forecast.html:** dự báo model đã chọn so với thực tế trên holdout test.

Mở HTML trong trình duyệt, chụp/xuất đồ thị thật và chèn vào Word với chú thích nguồn tệp.
Có thể chụp thêm trang **Đánh giá mô hình → Bằng chứng quá trình huấn luyện**.
Không cần đưa nguyên log dài vào thân báo cáo; chọn vài dòng minh họa và nộp file đầy đủ ở phụ lục.

Không kết luận model tốt chỉ vì loss train thấp: cần so validation/test, baseline naive và phạm vi dữ liệu.
Không báo cáo “độ chính xác = 100 − WAPE”. Không bắt buộc tuning phải cải thiện mọi thuật toán.

### Demo train trực tiếp trong buổi vấn đáp

Trong môi trường đã kích hoạt, tại thư mục gốc dự án:

```bash
python -m src.train --data data/raw/train.csv --output results/live_training --model-dir models/live_training --stores 1 --families BEVERAGES --quick --trials 1 --horizon 30
```

Lệnh này thực sự fit ba thuật toán và tuning trên CSV Kaggle, nhưng chỉ dùng **một cửa hàng × một nhóm**
và số cây/vòng giảm để trình diễn nhanh. Terminal in số mẫu, từng cấu hình, kết quả validation và test;
log được ghi ở results/live_training/training.log. Không ghi đè results/official hoặc models/official.
Sau demo nhỏ, quay lại kết quả chính thức 162 chuỗi để trình bày bảng so sánh trong báo cáo.
Không thay số liệu chính thức bằng số liệu của live demo.
Đã chạy thử lệnh trên: khoảng 4,91 giây trên máy thử nghiệm; máy khác có thể chậm hơn.

## 4. Những file nên nộp kèm

| File | Ý nghĩa |
| --- | --- |
| results/official/training.log | Đầu ra thực thi từng giai đoạn, tham số và thời gian |
| results/official/training_events.jsonl | Mỗi dòng là một sự kiện có timestamp; kiểm tra chọn model trước test |
| results/official/run_manifest.json | Input hash, nguồn dữ liệu, đặc trưng, mốc chia, phạm vi và phiên bản thư viện |
| results/official/metrics/tuning.csv | Cấu hình gốc và từng lượt tìm kiếm, tham số estimator đầy đủ |
| results/official/metrics/training_history.csv | Chẩn đoán HGB theo vòng boosting, từ mẫu train cố định |
| results/official/metrics/comparison.csv | Metric validation/test và tốc độ của các thuật toán |
| results/official/metrics/verification.json | Kiểm tra model nạp lại tái hiện test và dự báo 7/30 ngày |
| results/official/source_snapshot/ | Mã Python và requirements được sao chép tại thời điểm train |
| models/official/evaluation_model.joblib | Fit train+validation, dùng tái hiện test |
| models/official/model.joblib | Refit sau đánh giá trên toàn bộ lịch sử thuộc phạm vi đã chọn, dùng dự báo mới |

`.gitignore` loại .joblib khỏi Git, nhưng file vẫn có trong workspace.
Nếu nộp ZIP trực tiếp, đưa models và results vào ZIP; không đưa .venv, token hay dữ liệu trái điều khoản.
README có lệnh tái tạo, nên việc thiếu Git-tracked model không đồng nghĩa thiếu chương trình train.

## 5. Đoạn trình bày có thể sử dụng

> Nhóm xây dựng tập học từ lịch sử bán hàng Kaggle theo cửa hàng và nhóm hàng. Phạm vi thực nghiệm
> gồm 54 cửa hàng và ba nhóm BEVERAGES, GROCERY I, PRODUCE, được xác định trước khi lựa chọn mô hình.
> Sau khi tạo đặc trưng quá khứ và loại giai đoạn warm-up, tập train có 258.552 quan sát.
> Ba thuật toán Ridge, Random Forest và HistGradientBoosting được huấn luyện trên cùng tập dữ liệu.
> Mỗi cấu hình siêu tham số được fit độc lập và đánh giá bằng dự báo đệ quy 30 ngày trên validation.
> Mô hình được chọn theo WAPE validation, sau đó refit trên train+validation và đánh giá một lần
> trên test giữ riêng. Mọi lượt huấn luyện đều lưu thời điểm, tham số, thời gian và metric;
> model đánh giá được lưu để tái hiện kết quả, phân biệt với model triển khai refit sau đánh giá.

Chèn bảng metric, tên model thực sự được chọn và phân tích kết quả từ mục 9–10 báo cáo chính;
không điền trước người thắng khi quá trình train chưa hoàn tất.
