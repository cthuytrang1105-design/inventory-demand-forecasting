"""Quy tắc hỗ trợ quyết định; không tự đặt hàng, không tối ưu chi phí."""
import math
import numpy as np


def recommend_order(predictions, current_stock, lead_time, safety_stock=0, incoming=0,
                    same_unit=False, excess_factor=2.0):
    if not same_unit:
        raise ValueError("Cần xác nhận tồn kho và dự báo cùng đơn vị đo.")
    if isinstance(lead_time, bool) or not isinstance(lead_time, (int, np.integer)) or lead_time < 1:
        raise ValueError("Lead time phải là số nguyên dương.")
    demand = np.asarray(predictions, dtype=float)
    if demand.ndim != 1 or len(demand) < lead_time:
        raise ValueError("Lead time vượt chân trời dự báo; cần dự báo dài hơn.")
    if not np.isfinite(demand).all() or (demand < 0).any():
        raise ValueError("Dự báo phải hữu hạn và không âm.")
    if not all(np.isfinite(v) and v >= 0 for v in (current_stock, safety_stock, incoming)):
        raise ValueError("Tồn kho, tồn an toàn, hàng đang về phải hữu hạn và không âm.")
    if not np.isfinite(excess_factor) or excess_factor <= 1:
        raise ValueError("Hệ số dư thừa phải lớn hơn 1.")
    expected = float(demand[:lead_time].sum())
    available = float(current_stock + incoming)
    target = expected + safety_stock
    shortage = available < target
    # Ngưỡng dư thừa được công bố: A > 2 × T; khi T=0, A>0 là dư.
    excess = available > excess_factor * target
    return {"lead_time_demand": expected, "available_stock": available,
            "target_stock": target, "order_quantity": float(max(0, target - available)),
            "order_whole_units": math.ceil(max(0, target - available)),
            "shortage_risk": shortage, "excess_risk": excess,
            "status": "Nguy cơ thiếu hàng" if shortage else "Dư thừa theo quy tắc" if excess else "Trong ngưỡng mục tiêu"}
