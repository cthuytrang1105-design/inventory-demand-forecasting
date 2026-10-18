# BÁO CÁO GIỮA KỲ

**Đề tài:** Xây dựng hệ thống dự báo nhu cầu hàng hóa hỗ trợ tối ưu quản lý tồn kho bằng Machine Learning.

Người thực hiện: [điền danh sách thành viên]. Giảng viên: [điền thông tin xác nhận].
Bảng nhiệm vụ: [phan_cong_nhiem_vu.md](phan_cong_nhiem_vu.md).
Kết quả dưới đây được sinh trực tiếp bằng `python -m src.report --results results/official`.
Thời điểm thực nghiệm UTC: 2026-10-18T04:36:39.392768+00:00.

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

Nguồn lần chạy này: **Kaggle Store Sales – Time Series Forecasting; CSV do người dùng tải, schema đã kiểm tra tại chỗ**.
CSV đã kiểm tra thực tế có 3,000,888 dòng; dùng 272,808 dòng
cho 162 chuỗi. Cột thực tế: id, date, store_nbr, family, sales, onpromotion.
Đơn vị: sales tổng theo cửa hàng × nhóm hàng × ngày; có thể có phần thập phân, không phải đơn vị SKU đồng nhất.

Bộ dữ liệu tham chiếu là [Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
Tài liệu nguồn xác nhận train.csv gồm cửa hàng, nhóm hàng, sales và onpromotion;
sales có thể có phần thập phân. Lần chạy này dùng CSV người dùng cung cấp. Nguồn được công bố theo metadata người dùng; nhóm cần xác nhận đã tải đúng dữ liệu gốc và tuân thủ điều khoản sử dụng.
Phạm vi lần chạy: **54 cửa hàng** và các nhóm
**BEVERAGES, GROCERY I, PRODUCE**.
EDA toàn bộ CSV: 54 cửa hàng, 33 nhóm hàng, từ 2013-01-01 đến 2017-08-15. Phân tích lễ: Ghép đúng địa bàn; loại ngày transferred và Work Day.
Phạm vi mô hình có thể nhỏ hơn EDA; số liệu test chỉ đại diện các chuỗi thực sự được train.
Các tệp chính thức cần tải sau khi đăng nhập
và chấp thuận điều khoản; xem README để chạy toàn bộ quy trình trên dữ liệu Kaggle.

## 6. Phân tích và tiền xử lý dữ liệu

Thiếu theo cột trước xử lý: `{"id": 0, "date": 0, "store_nbr": 0, "family": 0, "sales": 0, "onpromotion": 0}`.
Số dòng trùng: 0; trùng khóa: 0.
Sales trong phạm vi mô hình có trung bình: 2504.0391, nhỏ nhất:
0.0000, lớn nhất: 124717.0000.

Chương trình từ chối ngày không hợp lệ, khóa trùng, sales thiếu/âm/vô hạn, family rỗng và
mã cửa hàng không nguyên dương. Thiếu onpromotion được điền 0 cho EDA; biến này không được dùng
trong mô hình nên không biến giả định điền thành thông tin dự báo.
Ngày thiếu trong mỗi chuỗi được chèn và forward fill từ quá khứ để lag có nghĩa theo ngày lịch.
Ngày chèn mang is_observed=False, tuyệt đối không làm nhãn train/test. Lần chạy này chèn
648 ngày và loại 4536 quan sát warm-up
không đủ 28 ngày lịch sử. Không dùng backward fill.

EDA HTML trong `results/official/figures/eda_*.html` gồm xu hướng ngày, tổng theo cửa hàng/nhóm hàng,
trung bình theo thứ/tháng và phân phối khi có/không khuyến mãi. Biểu đồ mô tả tương quan,
không chứng minh khuyến mãi gây tăng nhu cầu. Biểu đồ lễ chỉ sinh khi thực sự cung cấp
holidays_events.csv; nếu có stores.csv thì ghép National/Regional/Local đúng địa bàn,
loại ngày transferred và Work Day. Không tự gán lịch lễ ngoài dữ liệu.


### 6.1. Kết quả EDA trên toàn bộ CSV

Sales trung bình toàn CSV: 357.7757; median: 11.0000;
độ lệch chuẩn: 1101.9977; lớn nhất: 124717.0000.
Có 939,130 quan sát sales=0 (31.2951%). Phân phối lệch phải mạnh;
mean lớn hơn median nhiều, do nhiều nhóm có mức bán nhỏ/0 và một số nhóm có sales rất lớn.
Không coi mọi sales=0 là dữ liệu lỗi hoặc nhu cầu tiềm ẩn bằng 0 vì không có cột trạng thái hết hàng.
Những ngày không có quan sát trong toàn CSV: 2013-12-25, 2014-12-25, 2015-12-25, 2016-12-25.

Phân phối sales theo khuyến mãi, tính trên tất cả các quan sát:

| promotion | count | mean | q1 | median | q3 | lowerfence | upperfence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Có | 611329 | 1137.6937 | 121.3440 | 373.0000 | 1268.2490 | 0.0000 | 2988.6020 |
| Không | 2389559 | 158.2467 | 0.0000 | 3.0000 | 54.0000 | 0.0000 | 135.0000 |

Phân phối sales theo ngày lễ/sự kiện đã ghép đúng địa bàn:

| holiday | count | mean | q1 | median | q3 | lowerfence | upperfence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | 2754972 | 352.1305 | 0.0000 | 11.0000 | 193.0000 | 0.0000 | 482.5000 |
| True | 245916 | 421.0184 | 0.0000 | 13.0000 | 227.9528 | 0.0000 | 569.8480 |

Các khác biệt mô tả này có thể đồng thời chịu ảnh hưởng của quy mô nhóm hàng/cửa hàng và thời gian;
không kết luận tác động nhân quả. Các biến trên chưa được dùng trong mô hình dự báo hiện tại.
Các thống kê tổng theo ngày/cửa hàng/nhóm và trung bình theo thứ/tháng nằm trong
`results/official/metrics/eda_*.csv`; đồ thị tương ứng trong figures/eda_*.html.
Box plot xuất quartile/whisker từ toàn bộ dữ liệu thay vì đưa hàng triệu điểm vào HTML.


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

| Tập | start | end | rows |
| --- | --- | --- | --- |
| train | 2013-01-01 | 2017-06-16 | 263088 |
| validation | 2017-06-17 | 2017-07-16 | 4860 |
| test | 2017-07-17 | 2017-08-15 | 4860 |

Các mô hình có cùng tập quan sát train và cùng origin, horizon validation/test. Validation được
dự báo đệ quy từ cuối train; test từ cuối train+validation. Không nạp sales thật của ngày trước
trong holdout để tạo lag cho ngày kế tiếp. Các dự báo âm được chặn tại 0 cho mọi mô hình.
Mô hình tốt nhất được khóa trước khi chạy test. Test có ground truth được tách từ lịch sử;
test.csv Kaggle không có sales và không dùng làm nhãn đánh giá.


### 8.1. Minh chứng quá trình huấn luyện đã thực thi

Chương trình tạo **258,552 mẫu train** sau warm-up,
**21 cột đặc trưng trước one-hot** (2 khóa phân loại + các biến lịch/quá khứ).
Danh sách đặc trưng đầy đủ và SHA-256 dữ liệu/mã nguồn ghi trong run_manifest.json.
Lệnh tái lập đúng phạm vi lần chạy:

```bash
python -m src.train --data data/raw/train.csv --output results/official --model-dir models/official --horizon 30 --trials 3 --families BEVERAGES 'GROCERY I' PRODUCE
```

Đoạn mã thật thực hiện fit, tại `src/tuning.py`:

```python
model = build_model(name, params, quick)
model.fit(train_features[FEATURES], train_features.sales)
pred = recursive_forecast(model, history, horizon, name)
matched = join_actual(validation, pred)
record = metrics(matched.sales, matched.prediction)
```

Đoạn trích được rút gọn để trình bày; mã chạy đầy đủ còn đo thời gian, ghi nhật ký,
lưu tham số và chẩn đoán. Scaler/encoder được fit cùng model trong sklearn Pipeline.
Đây là việc học tham số từ X_train/y_train, khác với chỉ tải model đã huấn luyện để predict.
Ridge giải hồi quy có phạt L2; RF fit nhiều cây; HGB học tuần tự các vòng boosting.
Không có yêu cầu mọi thuật toán phải in epoch như mạng nơ-ron.

Nhật ký thực thi có timestamp UTC, không phải log minh họa:

| Bước thực thi | Timestamp UTC | Bằng chứng |
| --- | --- | --- |
| Fit Ridge cấu hình gốc hoàn tất | 2026-10-18T04:15:44.093059+00:00 | 0.134358 giây; validation WAPE 15.9947% |
| Khóa lựa chọn mô hình | 2026-10-18T04:31:24.221940+00:00 | RandomForest theo validation |
| Lần đánh giá test đầu tiên | 2026-10-18T04:31:24.812739+00:00 | Sau khi lựa chọn đã khóa |

Có **13 cấu hình được chấm** gồm cả seasonal naive; seasonal naive không fit.
Mỗi cấu hình ML ghi cặp trial_started/trial_completed, số mẫu, tham số đầy đủ,
train_seconds, inference_seconds và MAE/RMSE/WAPE validation. Không có nhãn test trong fit/tuning.

Các tệp nộp kèm để giảng viên đối chiếu:

- [training.log](../results/official/training.log): đầu ra thật của quá trình chạy.
- [training_events.jsonl](../results/official/training_events.jsonl): dấu vết từng bước với timestamp.
- [run_manifest.json](../results/official/run_manifest.json): phạm vi, mốc chia, môi trường, đặc trưng, SHA-256.
- [tuning.csv](../results/official/metrics/tuning.csv): từng cấu hình, tham số đầy đủ, kết quả đo thật.
- [source_snapshot](../results/official/source_snapshot): mã Python và requirements tại thời điểm train.

### 8.2. Hình và bảng minh họa quá trình train

1. [Sơ đồ chia thời gian](../results/official/figures/temporal_split.html): train đi trước validation/test.
2. [Đồ thị HGB theo vòng boosting](../results/official/figures/training_convergence.html): RMSE in-sample trên
   tối đa 5.000 mẫu train cách đều, cố định cho mọi cấu hình. Dùng chẩn đoán việc học,
   **không** coi loss train là chất lượng dự báo trên dữ liệu chưa thấy.
3. [Các lượt tuning và WAPE validation](../results/official/figures/tuning_validation.html): mỗi điểm là một
   model fit độc lập, không gọi các điểm này là epoch hoặc so như một chuỗi train liên tục.
4. Bảng trước/sau tuning ở mục 9 và bảng metric test ở mục 10.

Khi chuyển sang Word/PDF, chèn ảnh chụp các đồ thị hoặc trang Đánh giá mô hình đã chạy,
kèm chú thích nguồn tệp, phạm vi dữ liệu và đơn vị trục. Không tạo ảnh log giả.
Giữ file HTML/CSV/log trong gói mã nguồn để người chấm kiểm chứng.


## 9. Tối ưu siêu tham số

Sử dụng sklearn ParameterSampler, random_state=42, quick=False,
với số lượt cụ thể ghi trong `results/official/metrics/tuning.json`.
Mỗi cấu hình fit trên train và chấm bằng recursive validation một origin (không phải random K-fold).
Không gọi là RandomizedSearchCV: vòng tìm kiếm tự viết để bảo toàn semantics dự báo nhiều bước.
Không gian tìm kiếm: Ridge alpha 0.1/1/10/100/1000;
RF depth 6/10/16/None, min leaf 1/3/6, max features 0.7/1.0;
HGB learning rate 0.03/0.06/0.12, max leaves 7/15/31, L2 0/1/10.
Cấu hình mặc định cũng được chấm và có thể thắng; tuning không được giả định luôn cải thiện.
Toàn bộ kết quả và thời gian từng lượt nằm trong tuning.json và tuning.csv.

| model | before_MAE | best_MAE | before_WAPE_pct | best_WAPE_pct | best_params |
| --- | --- | --- | --- | --- | --- |
| SeasonalNaive | 517.8128 | 517.8128 | 14.5314 | 14.5314 | {} |
| Ridge | 569.9544 | 559.0898 | 15.9947 | 15.6898 | {"model__alpha": 1000.0} |
| RandomForest | 449.5331 | 449.0080 | 12.6153 | 12.6005 | {"model__min_samples_leaf": 6, "model__max_features": 0.7, "model__max_depth": 10} |
| HistGradientBoosting | 516.2127 | 478.7716 | 14.4865 | 13.4358 | {"model__max_leaf_nodes": 15, "model__learning_rate": 0.06, "model__l2_regularization": 1.0} |

## 10. Kết quả thực nghiệm và so sánh

**Chỉ có giá trị với nguồn dữ liệu ghi ở mục 5.** Thời gian đo bằng perf_counter trên
macOS-26.5.1-arm64-arm-64bit, Python 3.9.6,
scikit-learn 1.6.1, NumPy 1.26.4.
Native thread pool và RF n_jobs giới hạn 1 để phù hợp laptop. Timing phụ thuộc phần cứng và tải máy.

| model | validation_WAPE_pct | test_MAE | test_RMSE | test_WAPE_pct | train_seconds | inference_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| SeasonalNaive | 14.5314 | 517.5552 | 899.2306 | 14.9862 | 0.0000 | 0.2438 |
| Ridge | 15.6898 | 542.3826 | 797.5374 | 15.7051 | 0.1405 | 0.2866 |
| RandomForest | 12.6005 | 445.6334 | 692.4956 | 12.9037 | 148.3755 | 0.4393 |
| HistGradientBoosting | 13.4358 | 476.4709 | 732.5128 | 13.7966 | 13.6987 | 0.3129 |

MAE = mean(|y−ŷ|), RMSE = sqrt(mean((y−ŷ)²)), cùng đơn vị sales.
WAPE = 100 × sum(|y−ŷ|)/sum(|y|); mẫu số 0 thì trả None/không xác định.
WAPE toàn cục ưu tiên chuỗi có quy mô sales lớn, không phải trung bình phần trăm từng chuỗi.
`train_seconds` là refit trên train+validation để đánh giá test, không gồm tổng tuning;
`inference_seconds` gồm validation trong bộ nhớ, tạo đặc trưng và toàn bộ đệ quy test,
không gồm đọc đĩa/nạp model. Tổng lần chạy: 1151.051 giây.

Mô hình được chọn: **RandomForest**. RandomForest có WAPE validation nhỏ nhất (nếu WAPE không xác định dùng MAE); nếu bằng nhau ưu tiên thời gian suy luận thấp hơn. Test không tham gia tuning/chọn mô hình.
Tham số: `{"model__min_samples_leaf": 6, "model__max_features": 0.7, "model__max_depth": 10}`.
Tham số estimator đầy đủ (bao gồm mặc định):
`{"bootstrap": true, "ccp_alpha": 0.0, "criterion": "squared_error", "max_depth": 10, "max_features": 0.7, "max_leaf_nodes": null, "max_samples": null, "min_impurity_decrease": 0.0, "min_samples_leaf": 6, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "monotonic_cst": null, "n_estimators": 160, "n_jobs": 1, "oob_score": false, "random_state": 42, "verbose": 0, "warm_start": false}`.
Test MAE 445.6334, RMSE 692.4956, WAPE 12.9037%.
So với seasonal naive có test WAPE 14.9862%, mô hình được chọn có WAPE giảm 13.90% theo tỷ lệ tương đối. Đây là mức thay đổi WAPE, không phải accuracy. Kết luận chỉ áp dụng cho phạm vi và holdout của lần chạy này. Thuật toán có MAE test thấp nhất trong bảng là RandomForest. Không dùng kết quả test để thay đổi model hoặc tham số đã chọn bằng validation; nếu thứ hạng khác nhau, cần công bố khác biệt và kiểm tra thêm bằng backtest trong nghiên cứu tiếp theo.
Đây là kết quả đo, không phải cam kết độ chính xác khi vận hành.
Biểu đồ test: `results/official/figures/test_actual_vs_forecast.html`; sai số từng chuỗi: metrics/test_by_series.csv.

Artefact `evaluation_model.joblib` chỉ fit đến trước test để tái hiện đánh giá. Sau khi hoàn tất
đánh giá độc lập, `model.joblib` được refit trên toàn bộ lịch sử để dự báo mới. Không dùng bản
refit này để báo lại test. SHA-256 của CSV và danh sách chuỗi nằm trong run_manifest.json.

## 11. Mô tả ứng dụng

Streamlit gồm sáu trang: Tổng quan, Phân tích dữ liệu, Dự báo nhu cầu, Quản lý tồn kho,
Đánh giá mô hình, Giới thiệu hệ thống. Người dùng lọc cửa hàng/nhóm hàng, chọn 7 hoặc 30 ngày,
xem bảng/biểu đồ và tải CSV. Giao diện chỉ nạp model đã train, không tự kích hoạt huấn luyện.
Đầu vào thiếu model hoặc CSV sai được thông báo.
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

Kết quả chỉ đại diện cho CSV và các chuỗi được chọn trong lần chạy này; không mặc định suy rộng cho toàn bộ chuỗi hoặc thời kỳ khác. Chỉ một origin validation và một test,
horizon test của lần chạy là 30 ngày; chức năng 7/30 ngày không đồng nghĩa
đã đánh giá độc lập cả hai horizon. Chưa có khoảng dự báo hay backtest nhiều mùa.
Không dùng biến ngoại sinh nên có thể bỏ lỡ lễ/khuyến mãi/biến động kinh tế.
Forward fill ngày thiếu là giả định giữ nhu cầu gần nhất, có thể gây sai số; chỉ là state, không là nhãn.
Mức tổng hợp nhóm hàng có thể không đồng nhất đơn vị vật lý; muốn nhập hàng theo SKU cần dữ liệu chi tiết.
Quick mode tối đa 12 chuỗi nếu không ghi --max-series; kết quả chỉ đại diện chuỗi được chọn.

## 14. Hướng phát triển

Mở rộng số nhóm hàng được train, đối chiếu đơn vị và bổ sung rolling-origin backtest; thử LightGBM/XGBoost
trong cùng protocol; kết hợp lịch lễ theo địa bàn và kế hoạch khuyến mãi đã biết;
đánh giá theo từng nhóm hàng/horizon; bổ sung khoảng dự báo và theo dõi drift.
Để tối ưu tồn kho thực sự, cần dữ liệu tồn theo SKU, phân phối lead time, chi phí và ràng buộc vận hành.

## 15. Kết luận

Sản phẩm thực hiện đủ pipeline train–tuning–validation–test–lưu/nạp–suy luận,
so sánh ba thuật toán học máy và một mốc naive trên dữ liệu hiện có, có giao diện và quy tắc tồn kho.
Giá trị của bản giữa kỳ là quy trình tái lập, tránh leakage và công bố giới hạn dữ liệu.
Cần bổ sung đánh giá trên nhiều mốc thời gian độc lập trước khi kết luận chất lượng vận hành.
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
