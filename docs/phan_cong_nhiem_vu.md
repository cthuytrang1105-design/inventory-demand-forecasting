# Phân công nhiệm vụ

Chưa có danh sách thành viên được xác nhận. Để trống tên và mức đóng góp; nhóm điền theo công việc thực tế.
Các vai trò có thể do cùng một thành viên phụ trách nếu nhóm ít người.

| Thành viên | Vai trò | Nhiệm vụ | Sản phẩm phụ trách | Đóng góp thực tế | Xác nhận |
| --- | --- | --- | --- | --- | --- |
|  | Dữ liệu / EDA | Lấy CSV chính thức, kiểm tra schema, đơn vị, chất lượng, phân tích ngày lễ/khuyến mãi | src/data_processing.py, src/eda.py, biểu đồ |  |  |
|  | Feature engineering | Lag/rolling đúng ngày, tránh leakage, đồng nhất train/inference | src/feature_engineering.py, kiểm thử leakage |  |  |
|  | Machine Learning | Train ba thuật toán, tìm siêu tham số, chọn bằng validation, test giữ riêng | src/models.py, train.py, tuning.py, evaluate.py; artefact và metric |  |  |
|  | Pipeline / tồn kho | Đọc lịch sử mới, recursive forecast, công thức Q và kiểm tra đơn vị | src/predict.py, src/inventory.py |  |  |
|  | Giao diện / tích hợp | Sáu trang Streamlit, biểu đồ, CSV, trạng thái lỗi và nhãn nguồn | app.py, .streamlit/config.toml |  |  |
|  | Kiểm thử / tài liệu | Kiểm thử E2E, cập nhật số liệu, báo cáo và README, chuẩn bị vấn đáp | tests/, README.md, docs/ |  |  |

Mốc tham chiếu từ tài liệu giảng viên: nộp và nghiệm thu buổi học đầu tiên sau 26/10;
tài liệu không xác định năm/lịch học cụ thể. Nhóm tự xác nhận lịch, không tự đặt hạn nộp.
Thuyết trình/vấn đáp khoảng 20 phút; tài liệu nhóm không thay thế lịch môn học.
