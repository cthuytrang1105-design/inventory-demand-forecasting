# Hướng dẫn thuyết trình và vấn đáp (khoảng 20 phút)

| Thời lượng | Nội dung | Minh chứng |
| --- | --- | --- |
| 2 phút | Vấn đề dự báo nhóm hàng và rủi ro tồn kho; phân biệt với ERP | Giới thiệu và phạm vi |
| 3 phút | CSV Kaggle 3.000.888 dòng; phạm vi ML 54 cửa hàng × 3 nhóm; EDA, đơn vị sales | data/raw/train.metadata.json, trang Phân tích |
| 4 phút | Lag/rolling, shift trước rolling, chia theo ngày, recursive holdout | feature_engineering.py; temporal_split.html; mốc run_manifest |
| 4 phút | Chỉ dòng fit trong mã, log từng cấu hình; ba mô hình, random search và lý do chọn | training.log, training_convergence.html, tuning_summary.csv, trang Đánh giá |
| 3 phút | Thao tác dự báo 7/30 ngày, tải CSV, nhập tồn kho | Trang Dự báo / Quản lý tồn kho |
| 2 phút | Test, lưu/nạp model, hạn chế và hướng phát triển | docs/kiem_thu.md và JUnit XML |
| 2 phút | Vấn đáp | Các câu hỏi dưới đây |

Chuẩn bị trước: kích hoạt .venv, chạy các lệnh trong README, mở Streamlit, mở sẵn biểu đồ HTML
và các CSV kết quả trong results/official. Giao diện tự chọn nguồn Kaggle khi có artefact chính thức.
Chuẩn bị sẵn model trước buổi trình bày; có thể chạy một lượt fit nhỏ nếu còn thời gian.
Lệnh train trực tiếp phạm vi một chuỗi có trong docs/minh_chung_huan_luyen.md,
lưu riêng vào results/live_training và models/live_training.
Phân biệt EDA toàn bộ 33 nhóm với model chỉ học 3 nhóm ở 54 cửa hàng.
Nói rõ dữ liệu kết thúc năm 2017 nên dự báo bắt đầu 16/08/2017, không phải thời gian hiện tại.

Các câu hỏi cần giải thích được:

1. Vì sao không random split? Vì làm sai hướng thời gian và có thể dùng tương lai để dự báo quá khứ.
2. Vì sao shift trước rolling? Vì sales của chính ngày cần dự báo chưa có tại origin.
3. Ngày thứ hai dùng lag_1 nào? Dùng dự báo ngày thứ nhất, không dùng sales thật trong holdout.
4. Vì sao chọn mô hình? Chỉ dùng validation WAPE (MAE nếu mẫu số 0), test để báo cáo cuối cùng.
5. Tuning có luôn tốt hơn? Không. Cấu hình mặc định vẫn đủ điều kiện thắng, xem tuning_summary.csv.
6. HGB và RF khác nhau thế nào? RF tổng hợp cây học độc lập; boosting xây cây tuần tự sửa lỗi.
7. Tại sao hai artefact? evaluation_model tái hiện test; model triển khai refit sau đánh giá để dùng đủ lịch sử.
8. WAPE có hạn chế gì? Không xác định khi tổng sales 0; toàn cục thiên về chuỗi quy mô lớn.
9. Tồn kho lấy ở đâu? Người dùng nhập, phải xác nhận cùng đơn vị với sales.
10. Đã tối ưu chi phí chưa? Chưa; hiện chỉ dùng Q=max(0,D_L+S−C−I), chưa có dữ liệu chi phí.
11. Forecast là nhu cầu thật tuyệt đối? Không; sales có thể bị kiểm duyệt do hết hàng và gộp nhiều mặt hàng.
12. Tại sao chưa có LightGBM/XGBoost? Bản hiện tại dùng HGB của sklearn làm boosting để giữ cài đặt gọn;
    đáp ứng ba thuật toán thực sự, có thể mở rộng và so sánh trong cùng protocol.
13. Minh chứng đã train ở đâu? Chỉ `model.fit(...)` trong tuning.py; mở training.log và JSONL ghi
    thời điểm fit, train_rows, tham số đầy đủ, thời gian; đối chiếu tuning.csv và model.joblib.
    Đồ thị HGB là loss trên mẫu train, không dùng thay metric holdout.

Không tự điền tên thành viên, phần đóng góp, mốc học hoặc số liệu chưa được xác minh.
