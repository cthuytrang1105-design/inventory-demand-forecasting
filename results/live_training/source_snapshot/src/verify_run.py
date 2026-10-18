"""Tái kiểm chứng artefact bằng CSV thật, không fit lại hay lựa chọn lại mô hình."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data_processing import KEYS, load_sales
from src.evaluate import join_actual, metrics
from src.predict import forecast, load_artifact
from src.utils import file_hash, write_json


def verify_run(results="results/official"):
    root = Path(results)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    data_path = manifest["metadata"]["data_path"]
    if file_hash(data_path) != manifest["metadata"]["data_sha256"]:
        raise ValueError("CSV đã thay đổi so với lần train; không thể tái hiện holdout.")
    data = load_sales(data_path)
    data = data.merge(pd.DataFrame(manifest["series"], columns=KEYS), on=KEYS, how="inner", validate="many_to_one")
    cutoff = pd.Timestamp(manifest["split"]["test"]["start"])
    test_history = data[data.date < cutoff]
    test_actual = data[data.date >= cutoff]
    expected = pd.read_csv(root / "metrics/test_predictions.csv", parse_dates=["date"])
    with threadpool_limits(limits=1):
        evaluation_model = load_artifact(manifest["evaluation_model_path"])
        reproduced = join_actual(test_actual, forecast(evaluation_model, test_history, manifest["horizon"]))
        pd.testing.assert_frame_equal(expected, reproduced, check_dtype=False, atol=1e-8, rtol=1e-8)
        deployment = load_artifact(manifest["model_path"])
        outputs = {}
        for horizon in [7, 30]:
            prediction = forecast(deployment, data, horizon)
            if len(prediction) != horizon * len(manifest["series"]) or not np.isfinite(prediction.prediction).all():
                raise ValueError("Số dự báo không khớp hoặc không hữu hạn.")
            if prediction.duplicated(["date", *KEYS]).any():
                raise ValueError("Dự báo trùng khóa.")
            path = root / f"forecast_{horizon}_days_all_series.csv"
            prediction.to_csv(path, index=False)
            outputs[str(horizon)] = {"rows": len(prediction), "start": str(prediction.date.min().date()),
                                     "end": str(prediction.date.max().date()), "path": str(path)}
    report = {"verified": True, "simulated": manifest["metadata"]["simulated"],
              "input_sha256_matches": True, "holdout_predictions_reproduced": True,
              "test_rows": len(reproduced), "test_metrics_recomputed": metrics(reproduced.sales, reproduced.prediction),
              "evaluation_model_sha256": file_hash(manifest["evaluation_model_path"]),
              "deployment_model_sha256": file_hash(manifest["model_path"]), "forecast_checks": outputs}
    write_json(root / "metrics/verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/official")
    args = parser.parse_args()
    verify_run(args.results)
