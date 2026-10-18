# Dữ liệu trung gian

Đặc trưng được tạo trong bộ nhớ bởi `src.feature_engineering`; không lưu toàn bộ bảng lag
mặc định để tránh nhân bản CSV lớn. Báo cáo các hàng bị loại và ngày chèn được ghi ở
`results/<lần chạy>/metrics/data_quality.json`. Các ngày chèn không được làm nhãn train/test.
