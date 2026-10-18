# Hướng dẫn thuyết trình và vấn đáp (khoảng 20 phút)

| Thời lượng | Nội dung | Minh chứng |
| --- | --- | --- |
| 2 phút | Vấn đề dự báo nhóm hàng và rủi ro tồn kho; phân biệt với ERP | Giới thiệu và phạm vi |
| 3 phút | CSV thật đang dùng, nhãn mô phỏng, EDA, đơn vị sales | data/demo/sales.metadata.json, trang Phân tích |
| 4 phút | Lag/rolling, shift trước rolling, chia theo ngày, recursive holdout | feature_engineering.py; sơ đồ README; mốc run_manifest |
| 4 phút | Ba mô hình, random search, trước/sau, chọn bằng validation | comparison.csv, tuning_summary.csv, trang Đánh giá |
| 3 phút | Demo dự báo 7/30 ngày, tải CSV, nhập tồn kho | Trang Dự báo / Quản lý tồn kho |
| 2 phút | Test, lưu/nạp model, hạn chế và hướng phát triển | docs/kiem_thu.md và JUnit XML |
| 2 phút | Vấn đáp | Các câu hỏi dưới đây |

Chuẩn bị trước: kích hoạt .venv, chạy các lệnh trong README, mở Streamlit, mở sẵn biểu đồ HTML
và các CSV kết quả. Không train lại toàn bộ ngay khi trình bày. Kiểm tra nhãn mô phỏng còn hiển thị.
Nếu chưa có Kaggle, nói rõ chưa có thực nghiệm chính thức. Không gọi số liệu demo là độ chính xác trên Kaggle.

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

Không tự điền tên thành viên, phần đóng góp, mốc học hoặc số liệu chưa được xác minh.
