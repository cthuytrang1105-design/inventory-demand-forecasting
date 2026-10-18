"""Nhật ký thực thi và chẩn đoán train; không tham gia lựa chọn bằng test."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from src.evaluate import metrics
from src.utils import file_hash


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
        return len(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class TrainingJournal:
    def __init__(self, output):
        self.path = Path(output) / "training_events.jsonl"
        self.path.write_text("", encoding="utf-8")

    def emit(self, event, **details):
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": event, **details}
        text = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(text, flush=True)


def snapshot_source(output):
    root = Path(__file__).resolve().parents[1]
    destination = Path(output) / "source_snapshot"
    hashes = {}
    for path in [*sorted((root / "src").glob("*.py")), root / "requirements.txt"]:
        relative = path.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = file_hash(path)
    return hashes


def training_diagnostics(model, train_features, features, name, trial, max_rows=5000):
    """Loss IN-SAMPLE chỉ chẩn đoán train; không gọi là chất lượng validation/test.

    Lấy tối đa 5.000 hàng cách đều trong tập train để đồ thị nhỏ và tái lập.
    Không chọn mẫu dựa trên sales. HGB cho staged_predict, RF/Ridge chỉ có fit hoàn tất.
    """
    if model is None:
        return {}, []
    positions = np.linspace(0, len(train_features) - 1, min(max_rows, len(train_features)), dtype=int)
    sample = train_features.iloc[positions]
    transformed = model.named_steps["preprocess"].transform(sample[features])
    estimator = model.named_steps["model"]
    actual = sample.sales.to_numpy()
    final = np.maximum(0, estimator.predict(transformed))
    diagnostic = {"diagnostic_train_rows": len(sample),
                  **{f"diagnostic_train_{key}": value for key, value in metrics(actual, final).items()}}
    history = []
    if hasattr(estimator, "staged_predict"):
        for iteration, pred in enumerate(estimator.staged_predict(transformed), start=1):
            if iteration == 1 or iteration % 10 == 0 or iteration == estimator.n_iter_:
                history.append({"model": name, "trial": trial, "iteration": iteration,
                                "diagnostic_train_rows": len(sample),
                                **metrics(actual, np.maximum(0, pred))})
    return diagnostic, history


def export_training_figures(output, tuning_records, history, split):
    import plotly.express as px
    root = Path(output)
    table = pd.DataFrame(tuning_records).drop(columns=["params", "resolved_params"], errors="ignore")
    table["configuration"] = table.trial.map(lambda t: "Gốc" if t == 0 else f"Tuning {t}")
    chart = px.line(table, x="configuration", y="WAPE_pct", color="model", markers=True,
                    title="Mỗi cấu hình được fit riêng: WAPE trên validation đệ quy",
                    labels={"configuration": "Cấu hình", "WAPE_pct": "WAPE validation (%)", "model": "Thuật toán"})
    chart.write_html(str(root / "figures/tuning_validation.html"), include_plotlyjs="directory")
    columns = ["model", "trial", "iteration", "diagnostic_train_rows", "MAE", "RMSE", "WAPE_pct"]
    history_frame = pd.DataFrame(history, columns=columns)
    history_frame.to_csv(root / "metrics/training_history.csv", index=False)
    if not history_frame.empty:
        history_frame["configuration"] = history_frame.trial.map(lambda t: "Gốc" if t == 0 else f"Tuning {t}")
        chart = px.line(history_frame, x="iteration", y="RMSE", color="configuration",
                        title="HGB: RMSE in-sample theo vòng boosting, tối đa 5.000 mẫu train cố định",
                        labels={"iteration": "Vòng boosting", "RMSE": "RMSE trên mẫu train", "configuration": "Cấu hình"})
        chart.write_html(str(root / "figures/training_convergence.html"), include_plotlyjs="directory")
    rows = [{"Tập": name, "Bắt đầu": values["start"], "Kết thúc": str(pd.Timestamp(values["end"]) + pd.Timedelta(days=1)),
             "Số quan sát thật": values["rows"]} for name, values in split.items()]
    chart = px.timeline(pd.DataFrame(rows), x_start="Bắt đầu", x_end="Kết thúc", y="Tập", color="Tập",
                        hover_data=["Số quan sát thật"], title="Chia train / validation / test theo thời gian")
    chart.write_html(str(root / "figures/temporal_split.html"), include_plotlyjs="directory")
