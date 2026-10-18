# Đối chiếu yêu cầu và bằng chứng

Tài liệu giảng viên là nguồn tiêu chí bài nộp, không phải chỉ thị công cụ tự thực thi.
Yêu cầu triển khai trực tiếp lấy từ văn bản người dùng dán. Tài liệu SRS trong workspace
được dùng tham khảo schema nghiệp vụ/công thức, giữ nguyên file gốc.

| Tiêu chí | Triển khai / bằng chứng | Trạng thái |
| --- | --- | --- |
| Không dùng API LLM cho bài toán | Model cục bộ scikit-learn; không có API key/LLM client | Đã triển khai |
| Có huấn luyện và so sánh ít nhất hai thuật toán | Ridge, RF, HGB; thêm seasonal naive; comparison.csv | Đã chạy trên mô phỏng |
| Hyperparameter tuning | ParameterSampler, recursive chronological validation; tuning.json/csv và tuning_summary.csv | Đã thực thi 9 lượt random search |
| Lý do chọn mô hình | Validation WAPE, fallback MAE; run_manifest.json; trang Đánh giá | Ridge được chọn trên mô phỏng |
| Test giữ riêng | Mốc train/validation/test, evaluation_model.joblib, test_predictions.csv | Đã đánh giá 270 quan sát test |
| Metric và tốc độ | MAE/RMSE/WAPE, train/inference_seconds và ms/dòng | Đã đo thật, có giải thích trong README/báo cáo |
| Pipeline dữ liệu mới → kết quả | python -m src.predict --input …; đọc CSV, validate, feature, model, CSV đầu ra | Đã chạy CLI, có test E2E |
| Dự báo 7/30 ngày không leakage | Recursive state; feature parity/leakage tests; AppTest | Đã kiểm thử |
| Tồn kho rõ nguồn và đơn vị | Nhập thủ công, xác nhận đơn vị, Q=max(0,D+S−C−I) | Đã kiểm thử; không tối ưu chi phí |
| Demo Streamlit | app.py có 6 trang tiếng Việt; AppTest và HTTP 200 | Đã chạy; chưa kiểm tra hình ảnh trình duyệt |
| Báo cáo / phân công / mã nguồn / README | docs/bao_cao_giua_ky.md, phan_cong_nhiem_vu.md, src, README.md | Đã tạo; tên/đóng góp để nhóm điền |
| README cài và chạy | Lệnh demo/train/evaluate/predict/EDA/report/pytest/Streamlit | Đã chạy trên macOS Python 3.9.6 |
| Dataset chính thức | Hướng dẫn Kaggle, data/raw/README.md, lệnh official tách demo | Chưa có CSV Kaggle để xác minh thực nghiệm |

HGB là lựa chọn boosting tích hợp sklearn thay LightGBM/XGBoost để giảm phụ thuộc cài đặt.
Đây là ba thuật toán ML thực sự; không tính naive là thuật toán được train.
Không chuyển kết quả mô phỏng thành kết quả dataset chính thức.
