import numpy as np
import pandas as pd
import pytest
from src.evaluate import metrics
from src.inventory import recommend_order
from src.predict import forecast, load_artifact, recursive_forecast


class LagPlusOne:
    def predict(self, features):
        return features.lag_1.to_numpy() + 1


def test_recursive_forecast_uses_own_predictions(sample):
    history = sample[(sample.store_nbr == 1) & (sample.family == "BEVERAGES")]
    pred = recursive_forecast(LagPlusOne(), history, 30)
    np.testing.assert_allclose(pred.prediction, history.sales.iloc[-1] + np.arange(1, 31))
    assert pred.date.min() == history.date.max() + pd.Timedelta(days=1)
    assert len(pred) == 30 and pred.date.is_monotonic_increasing


def test_seasonal_naive_recursion(sample):
    history = sample[(sample.store_nbr == 1) & (sample.family == "BEVERAGES")]
    pred = recursive_forecast(None, history, 30, "SeasonalNaive")
    np.testing.assert_allclose(pred.prediction, np.resize(history.sales.tail(7).to_numpy(), 30))


def test_forecast_bad_history(sample, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "missing.joblib")
    artifact = {"series": [[99, "UNKNOWN"]], "model": LagPlusOne(), "model_name": "test"}
    with pytest.raises(ValueError, match="chưa được huấn luyện"):
        forecast(artifact, sample)
    with pytest.raises(ValueError, match="28 ngày"):
        recursive_forecast(LagPlusOne(), sample[sample.date > sample.date.max() - pd.Timedelta(days=10)])
    stale = sample.drop(sample[(sample.store_nbr == 1) & (sample.date == sample.date.max())].index)
    with pytest.raises(ValueError, match="chưa cập nhật"):
        recursive_forecast(LagPlusOne(), stale)


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True, 366])
def test_invalid_horizon(sample, horizon):
    with pytest.raises(ValueError):
        recursive_forecast(LagPlusOne(), sample, horizon)


def test_inventory_srs_example():
    result = recommend_order([40, 45, 50, 45, 50], 120, 5, 30, 20, same_unit=True)
    assert result["lead_time_demand"] == 230
    assert result["available_stock"] == 140
    assert result["target_stock"] == 260
    assert result["order_quantity"] == 120
    assert result["shortage_risk"] and not result["excess_risk"]
    assert recommend_order([1] * 7, 500, 5, same_unit=True)["excess_risk"]


@pytest.mark.parametrize("kwargs", [{"current_stock": -1}, {"lead_time": 0}, {"lead_time": 8},
                                   {"safety_stock": np.nan}, {"incoming": -1}, {"same_unit": False}])
def test_invalid_inventory(kwargs):
    inputs = dict(predictions=[10] * 7, current_stock=10, lead_time=5, same_unit=True)
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        recommend_order(**inputs)


def test_metrics_zero_denominator():
    assert metrics([0, 0], [1, 0])["WAPE_pct"] is None
    got = metrics([10, 20], [12, 18])
    assert got["MAE"] == 2 and got["RMSE"] == 2
    assert got["WAPE_pct"] == pytest.approx(100 * 4 / 30)
