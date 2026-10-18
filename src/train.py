"""Huấn luyện, random search, chọn theo validation, test giữ riêng, lưu artefact."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data_processing import KEYS, load_sales, quality_summary, split_dates
from src.evaluate import export_test_figure, join_actual, metrics
from src.feature_engineering import FEATURES, make_features
from src.models import build_model
from src.predict import forecast, load_artifact, recursive_forecast
from src.tuning import SEARCH_SPACES, score_key, search_model
from src.utils import file_hash, read_metadata, write_json

MODEL_NAMES = ["SeasonalNaive", "Ridge", "RandomForest", "HistGradientBoosting"]


def train_run(data_path, output="results/demo", model_dir="models/demo", horizon=30,
              trials=3, quick=False, max_series=None, stores=None, families=None):
    output, model_dir = Path(output), Path(model_dir)
    for folder in (output / "metrics", output / "figures", model_dir):
        folder.mkdir(parents=True, exist_ok=True)
    started_run = time.perf_counter()
    raw = pd.read_csv(data_path)
    quality = quality_summary(raw)
    df = load_sales(data_path)
    if stores:
        df = df[df.store_nbr.isin(stores)]
    if families:
        df = df[df.family.isin(families)]
    if df.empty:
        raise ValueError("Bộ lọc không còn dữ liệu.")
    if max_series is None and quick:
        max_series = 12
    if max_series is not None:
        if max_series < 1:
            raise ValueError("max_series phải >= 1.")
        selected = df[KEYS].drop_duplicates().sort_values(KEYS).head(max_series)
        df = df.merge(selected, on=KEYS, validate="many_to_one").sort_values([*KEYS, "date"])
    train, validation, test = split_dates(df, horizon)
    # Mỗi chuỗi phải tồn tại đủ lịch sử và có cùng mốc chốt ở mọi origin.
    for partition in (train, pd.concat([train, validation]), df):
        for key, group in partition.groupby(KEYS):
            if group.date.max() != partition.date.max() or (group.date.max() - group.date.min()).days < 59:
                raise ValueError(f"Chuỗi {key} không có ít nhất 60 ngày và cùng ngày cuối; hãy lọc chuỗi phù hợp.")
    counts = [len(part[KEYS].drop_duplicates()) for part in (train, validation, test)]
    if len(set(counts)) != 1:
        raise ValueError("Có chuỗi mới/mất chuỗi giữa train, validation, test.")
    train_features = make_features(train)
    train_features = train_features[train_features.is_observed].dropna(subset=FEATURES)
    if train_features.empty:
        raise ValueError("Không còn mẫu train sau tạo lag.")
    tuning_records, best_models = [], []
    # Giới hạn native BLAS/OpenMP để laptop ổn định và timing có thể so sánh.
    with threadpool_limits(limits=1):
        for name in MODEL_NAMES:
            print(f"Huấn luyện / tuning {name}...", flush=True)
            best, records = search_model(name, train_features, train, validation, trials, quick)
            tuning_records.extend(records)
            best_models.append(best)
        selected = min(best_models, key=score_key)
        print(f"Chọn bằng validation: {selected['model']} ({selected['WAPE_pct']}% WAPE)", flush=True)
        trainval = pd.concat([train, validation], ignore_index=True)
        trainval_features = make_features(trainval)
        trainval_features = trainval_features[trainval_features.is_observed].dropna(subset=FEATURES)
        comparison, all_test, selected_eval_artifact = [], [], None
        series = [[int(s), str(f)] for s, f in df[KEYS].drop_duplicates().itertuples(index=False, name=None)]
        meta = {**read_metadata(data_path), "data_sha256": file_hash(data_path),
                "data_path": str(Path(data_path)), "scope": "cửa hàng × nhóm hàng × ngày",
                "series_count": len(series)}
        # Test chỉ chạy sau khi đã khóa lựa chọn theo validation.
        for best in best_models:
            name = best["model"]
            model = build_model(name, best["params"], quick)
            started = time.perf_counter()
            if model is not None:
                model.fit(trainval_features[FEATURES], trainval_features.sales)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            pred = recursive_forecast(model, trainval, horizon, name)
            inference_seconds = time.perf_counter() - started
            matched = join_actual(test, pred)
            all_test.append(matched.assign(model=name))
            comparison.append({"model": name, "selected": name == selected["model"],
                               **{f"validation_{k}": best[k] for k in ["MAE", "RMSE", "WAPE_pct"]},
                               **{f"test_{k}": v for k, v in metrics(matched.sales, matched.prediction).items()},
                               "train_seconds": fit_seconds, "inference_seconds": inference_seconds,
                               "inference_ms_per_row": 1000 * inference_seconds / len(pred),
                               "params": json.dumps(best["params"], ensure_ascii=False)})
            if name == selected["model"]:
                selected_eval_artifact = {"format_version": 1, "model_name": name, "model": model,
                                          "series": series, "metadata": meta,
                                          "training_end": str(trainval.date.max().date()), "purpose": "holdout_evaluation"}
                matched.to_csv(output / "metrics/test_predictions.csv", index=False)
                export_test_figure(matched, output / "figures/test_actual_vs_forecast.html")
                # Truy vết nhóm dễ/sai nhiều, không dùng để chọn model.
                group_rows = []
                for key, group in matched.groupby(KEYS):
                    group_rows.append(dict(zip(KEYS, key), **metrics(group.sales, group.prediction)))
                pd.DataFrame(group_rows).to_csv(output / "metrics/test_by_series.csv", index=False)
        joblib.dump(selected_eval_artifact, model_dir / "evaluation_model.joblib")
        # Refit mô hình triển khai trên TẤT CẢ nhãn đã biết, sau đánh giá độc lập.
        final_model = build_model(selected["model"], selected["params"], quick)
        full_features = make_features(df)
        observed = full_features[full_features.is_observed].dropna(subset=FEATURES)
        started = time.perf_counter()
        if final_model is not None:
            final_model.fit(observed[FEATURES], observed.sales)
        deployment_fit = time.perf_counter() - started
        artifact = {"format_version": 1, "model_name": selected["model"], "model": final_model,
                    "series": series, "metadata": meta, "training_end": str(df.date.max().date()),
                    "purpose": "deployment_refit_after_test", "best_params": selected["params"]}
        joblib.dump(artifact, model_dir / "model.joblib")
        # Kiểm tra lưu/nạp và chạy dữ liệu qua API inference thật.
        reloaded = load_artifact(model_dir / "model.joblib")
        future = forecast(reloaded, df, 7)
        original = forecast(artifact, df, 7)
        pd.testing.assert_frame_equal(future, original)
        future.to_csv(output / "forecast_7_days.csv", index=False)
    pd.DataFrame(comparison).to_csv(output / "metrics/comparison.csv", index=False)
    pd.concat(all_test, ignore_index=True).to_csv(output / "metrics/all_models_test_predictions.csv", index=False)
    write_json(output / "metrics/tuning.json", {"search_spaces": SEARCH_SPACES,
               "strategy": "ParameterSampler + chronological single-origin recursive validation; default also eligible",
               "trials_per_model": trials, "records": tuning_records})
    json_rows = [{k: json.dumps(v, ensure_ascii=False) if k == "params" else v for k, v in row.items()} for row in tuning_records]
    pd.DataFrame(json_rows).to_csv(output / "metrics/tuning.csv", index=False)
    summary = []
    for best in best_models:
        before = next(r for r in tuning_records if r["model"] == best["model"] and r["stage"] == "before_tuning")
        summary.append({"model": best["model"], "before_MAE": before["MAE"], "best_MAE": best["MAE"],
                        "before_WAPE_pct": before["WAPE_pct"], "best_WAPE_pct": best["WAPE_pct"],
                        "best_params": json.dumps(best["params"], ensure_ascii=False)})
    pd.DataFrame(summary).to_csv(output / "metrics/tuning_summary.csv", index=False)
    quality.update({"selected_rows": len(df), "inserted_calendar_days": int((~full_features.is_observed).sum()),
                    "excluded_warmup_observed_rows": len(df) - len(observed),
                    "sales_summary": {k: float(v) for k, v in df.sales.describe().items()}})
    write_json(output / "metrics/data_quality.json", quality)
    reason = (f"{selected['model']} có WAPE validation nhỏ nhất (nếu WAPE không xác định dùng MAE); "
              "nếu bằng nhau ưu tiên thời gian suy luận thấp hơn. Test không tham gia tuning/chọn mô hình.")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "metadata": meta,
                "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                **{p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib", "streamlit"]}},
                "quick": quick, "max_series": max_series, "horizon": horizon, "seed": 42,
                "selected_model": selected["model"], "best_params": selected["params"],
                "selection_reason": reason, "series": series,
                "split": {name: {"start": str(part.date.min().date()), "end": str(part.date.max().date()), "rows": len(part)}
                          for name, part in [("train", train), ("validation", validation), ("test", test)]},
                "train_feature_rows": len(train_features), "test_untouched_during_selection": True,
                "deployment_refit_seconds": deployment_fit, "total_seconds": time.perf_counter() - started_run,
                "model_path": str(model_dir / "model.joblib"),
                "evaluation_model_path": str(model_dir / "evaluation_model.joblib"),
                "assumptions": ["Không dùng biến ngoại sinh, không cần biết khuyến mãi/dầu/giao dịch tương lai.",
                                "Ngày thiếu ffill từ quá khứ chỉ làm state, không tạo nhãn train/test.",
                                "Chỉ đánh giá quan sát thật; dự báo nhiều bước recursive, chặn âm tại 0.",
                                "Một origin validation và một origin test, chưa rolling-origin nhiều giai đoạn.",
                                "Artefact triển khai refit trên cả test sau khi test đã hoàn tất; tái hiện test bằng evaluation_model."]}
    write_json(output / "run_manifest.json", manifest)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="results/demo")
    p.add_argument("--model-dir", default="models/demo")
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--quick", action="store_true", help="Giảm số cây/vòng boosting; tối đa 12 chuỗi nếu không chỉ định")
    p.add_argument("--max-series", type=int)
    p.add_argument("--stores", nargs="+", type=int)
    p.add_argument("--families", nargs="+")
    args = p.parse_args()
    train_run(args.data, args.output, args.model_dir, args.horizon, args.trials,
              args.quick, args.max_series, args.stores, args.families)


if __name__ == "__main__":
    main()
