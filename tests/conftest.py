import pytest
from src.demo_data import generate_demo
from src.data_processing import load_sales


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "sales.csv"
    generate_demo(path, days=150)
    return load_sales(path)
