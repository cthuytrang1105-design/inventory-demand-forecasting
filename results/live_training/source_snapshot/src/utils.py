"""Tiện ích lưu kết quả có thể truy vết."""
import hashlib
import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(path):
    sidecar = Path(path).with_suffix(".metadata.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    return {"source": "CSV do người dùng cung cấp; chưa xác minh nguồn", "simulated": False,
            "unit": "sales — cần xác nhận đơn vị trước khi tính tồn kho"}
