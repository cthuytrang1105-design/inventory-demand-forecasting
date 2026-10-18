# Dữ liệu Kaggle (chưa được cung cấp trong workspace)

Tải từ [Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
sau khi đăng nhập và chấp thuận điều khoản cuộc thi. Giải nén `train.csv` vào thư mục này.
Các tệp bổ sung tùy chọn: `stores.csv`, `holidays_events.csv` để phân tích EDA.

Schema tối thiểu của lịch sử: `date,store_nbr,family,sales`.
`onpromotion` là cột tùy chọn, chỉ dùng trong EDA ở phiên bản hiện tại.
`id` được giữ để truy vết nhưng không làm đặc trưng.

`test.csv` Kaggle không có sales, do đó **không phải CSV lịch sử** cho `src.predict`.
Test đánh giá nội bộ được tách ở cuối `train.csv` để có ground truth.
Chương trình kiểm tra cột, ngày, trùng khóa, giá trị không hợp lệ trước huấn luyện.
Không có cột tồn kho trong schema này. `sales` có thể có phần thập phân và là tổng theo nhóm hàng;
không tự suy diễn là số SKU, số thùng hoặc doanh thu tiền tệ.

Có thể thêm `train.metadata.json` để ghi nguồn do nhóm tự xác nhận:

```json
{
  "source": "Kaggle Store Sales — CSV gốc do nhóm tải và đối chiếu",
  "simulated": false,
  "unit": "sales theo nhóm sản phẩm; cần kiểm tra/quy đổi trước sử dụng tồn kho thực"
}
```

Không tải dữ liệu từ nguồn bên thứ ba không rõ nguồn gốc; không đưa Kaggle token vào repo.
