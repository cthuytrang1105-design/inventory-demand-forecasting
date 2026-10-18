import json
import subprocess
import sys
import numpy as np
import pandas as pd
from src.data_processing import load_sales
from src.demo_data import generate_demo
from src.evaluate import join_actual, metrics
from src.predict import forecast, load_artifact
from src.train import MODEL_NAMES, train_run


def test_complete_pipeline_new_csv_reload_and_holdout(tmp_path):
    path, results, models = tmp_path / "sales.csv", tmp_path / "results", tmp_path / "models"
    generate_demo(path, days=150)
    manifest = train_run(path, results, models, horizon=7, trials=1, quick=True, max_series=2)
    assert manifest["test_untouched_during_selection"]
    assert manifest["metadata"]["simulated"]
    tuning = json.loads((results / "metrics/tuning.json").read_text())["records"]
    assert len([r for r in tuning if r["stage"] == "random_search"]) == 3
    comparison = pd.read_csv(results / "metrics/comparison.csv")
    assert set(comparison.model) == set(MODEL_NAMES)
    assert comparison.selected.sum() == 1
    data = load_sales(path).merge(pd.DataFrame(manifest["series"], columns=["store_nbr", "family"]))
    # Đọc một CSV mới, không dùng object giữ trong train.
    new_path = tmp_path / "new_history.csv"
    data.sample(frac=1, random_state=7).to_csv(new_path, index=False)
    loaded = load_artifact(models / "model.joblib")
    result = forecast(loaded, load_sales(new_path), 7)
    assert len(result) == 14 and np.isfinite(result.prediction).all()
    pd.testing.assert_frame_equal(result, forecast(load_artifact(models / "model.joblib"), data, 7))
    # Chứng minh artefact đánh giá tái tạo dự báo test, không dùng bản refit toàn bộ.
    evaluation = load_artifact(models / "evaluation_model.joblib")
    test_start = pd.Timestamp(manifest["split"]["test"]["start"])
    expected = pd.read_csv(results / "metrics/test_predictions.csv", parse_dates=["date"])
    reproduced = join_actual(data[data.date >= test_start], forecast(evaluation, data[data.date < test_start], 7))
    pd.testing.assert_frame_equal(expected, reproduced, check_dtype=False, atol=1e-8)
    selected = comparison[comparison.selected].iloc[0]
    assert abs(metrics(reproduced.sales, reproduced.prediction)["MAE"] - selected.test_MAE) < 1e-8
    output = tmp_path / "cli_forecast.csv"
    process = subprocess.run([sys.executable, "-m", "src.predict", "--input", str(new_path),
                              "--model", str(models / "model.joblib"), "--horizon", "7",
                              "--output", str(output)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert len(pd.read_csv(output)) == 14
