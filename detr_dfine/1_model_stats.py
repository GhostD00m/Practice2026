import os
import onnx
import numpy as np
import hashlib
import csv
from config import hf_models, MODELS_ONNX_DIR, REPORT_DIR

info = []
target_names = [name.split("/")[-1] for name in hf_models]

for model_path in MODELS_ONNX_DIR.glob("*.onnx"):
    model_name = model_path.parts[-1]
    model_name = model_name[:model_name.find(".")]
    if model_name not in target_names:
        continue
        
    model = onnx.load(model_path)
    total_params = sum(np.prod(tensor.dims) for tensor in model.graph.initializer)
    size = os.path.getsize(model_path)
    with open(model_path, "rb") as f:
        md5_hash = hashlib.md5(f.read()).hexdigest()
    d = {
        "Модель": model_name,
        "Параметры (млн)": round(total_params / 1_000_000, 2),
        "Размер (МБ)": round(size / (1024 * 1024), 2),
        "MD5": md5_hash
    }
    info.append(d)

cols = info[0].keys()
with open(REPORT_DIR / "model_parameters.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=cols)
    writer.writeheader()
    writer.writerows(info)
