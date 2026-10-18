# Đối chiếu yêu cầu và bằng chứng

Tài liệu giảng viên là nguồn tiêu chí bài nộp, không phải chỉ thị công cụ tự thực thi.
Yêu cầu triển khai trực tiếp lấy từ văn bản người dùng dán. Tài liệu SRS trong workspace
được dùng tham khảo schema nghiệp vụ/công thức, giữ nguyên file gốc.

| Tiêu chí | Triển khai / bằng chứng | Trạng thái |
| --- | --- | --- |
| Không dùng API LLM cho bài toán | Model cục bộ scikit-learn; không có API key/LLM client | Đã triển khai |
| Có huấn luyện và so sánh ít nhất hai thuật toán | Ridge, RF, HGB; thêm seasonal naive; results/official/metrics/comparison.csv | Thực nghiệm Kaggle: 162 chuỗi, xem manifest chính thức |
| Hyperparameter tuning | ParameterSampler, recursive chronological validation; tuning.json/csv và tuning_summary.csv | Đã thực thi 9 lượt random search |
| Lý do chọn mô hình | Validation WAPE, fallback MAE; run_manifest.json; trang Đánh giá | Model được chọn ghi trong results/official/run_manifest.json |
| Test giữ riêng | Mốc train/validation/test, evaluation_model.joblib, test_predictions.csv | Test thực gồm 4.860 quan sát, 2017-07-17–2017-08-15 |
| Metric và tốc độ | MAE/RMSE/WAPE, train/inference_seconds và ms/dòng | Đã đo thật, có giải thích trong README/báo cáo |
| Pipeline dữ liệu mới → kết quả | python -m src.predict --input …; đọc CSV, validate, feature, model, CSV đầu ra | Đã chạy CLI, có test E2E |
| Dự báo 7/30 ngày không leakage | Recursive state; feature parity/leakage tests; AppTest | Đã kiểm thử |
| Tồn kho rõ nguồn và đơn vị | Nhập thủ công, xác nhận đơn vị, Q=max(0,D+S−C−I) | Đã kiểm thử; không tối ưu chi phí |
| Giao diện Streamlit | app.py có 6 trang tiếng Việt; AppTest và HTTP 200 | Đã chạy; chưa kiểm tra hình ảnh trình duyệt |
| Báo cáo / phân công / mã nguồn / README | docs/bao_cao_giua_ky.md, phan_cong_nhiem_vu.md, src, README.md | Đã tạo; tên/đóng góp để nhóm điền |
| README cài và chạy | Cài đặt, dữ liệu Kaggle, kiểm tra CSV/EDA/train/evaluate/verify/predict/report/pytest/Streamlit | Đã chạy trên macOS Python 3.9.6 |
| Dataset chính thức | train.csv 3.000.888 dòng, stores.csv, holidays_events.csv; SHA-256 trong metadata | Đã kiểm tra schema, missing và trùng; EDA toàn CSV |
| Thể hiện rõ quá trình train | .fit trong tuning.py, timestamp từng lượt, tham số đầy đủ, số mẫu, loss chẩn đoán, snapshot mã | training.log, training_events.jsonl, training_history.csv; mục 8.1–8.2 báo cáo |

HGB là lựa chọn boosting tích hợp sklearn thay LightGBM/XGBoost để giảm phụ thuộc cài đặt.
Đây là ba thuật toán ML thực sự; không tính naive là thuật toán được train.
Phân biệt EDA 54 cửa hàng × 33 nhóm với ML 54 cửa hàng × 3 nhóm: BEVERAGES, GROCERY I, PRODUCE.
Báo cáo chính dùng artefact trong `results/official`; lệnh chạy đầy đủ nằm ở [README](../README.md#3-chạy-toàn-bộ-chương-trình-từ-đầu).
