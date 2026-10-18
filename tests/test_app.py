from pathlib import Path
import shutil
import pytest
from streamlit.testing.v1 import AppTest
from src.demo_data import generate_demo
from src.train import train_run


@pytest.fixture(scope="module")
def app_file(tmp_path_factory):
    root = tmp_path_factory.mktemp("streamlit")
    data = root / "data/demo/sales.csv"
    generate_demo(data, days=150)
    train_run(data, root / "results/demo", root / "models/demo", horizon=7, trials=1,
              quick=True, max_series=2)
    shutil.copyfile(Path(__file__).resolve().parents[1] / "app.py", root / "app.py")
    return str(root / "app.py")


@pytest.mark.parametrize("page", ["Tổng quan", "Phân tích dữ liệu", "Dự báo nhu cầu",
                                  "Quản lý tồn kho", "Đánh giá mô hình", "Giới thiệu hệ thống"])
def test_all_six_pages(app_file, page):
    at = AppTest.from_file(app_file, default_timeout=20).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception
    assert at.title[0].value == page


def test_forecast_and_inventory_interaction(app_file):
    at = AppTest.from_file(app_file, default_timeout=20).run()
    at.sidebar.radio[0].set_value("Dự báo nhu cầu").run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert len(at.dataframe[0].value) == 7
    # Thay horizon không hiển thị dự báo cũ.
    next(widget for widget in at.selectbox if widget.label == "Số ngày dự báo").set_value(30).run()
    assert len(at.dataframe) == 0
    at.button[0].click().run()
    assert len(at.dataframe[0].value) == 30
    at.sidebar.radio[0].set_value("Quản lý tồn kho").run()
    at.button[0].click().run()
    assert at.error and "đơn vị" in at.error[0].value
    at.checkbox[0].check().run()
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert len(at.metric) == 3


def test_missing_official_artifacts_message(app_file):
    at = AppTest.from_file(app_file, default_timeout=20).run()
    at.sidebar.selectbox[0].set_value("CSV người dùng / Kaggle").run()
    at.sidebar.radio[0].set_value("Dự báo nhu cầu").run()
    assert not at.exception
    assert at.error
