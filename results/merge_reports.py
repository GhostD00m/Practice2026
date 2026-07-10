import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent

df_detr_metrics = pd.read_csv(RESULTS_DIR / "detr_dfine_metrics.csv")
df_yolo_metrics = pd.read_csv(RESULTS_DIR / "yolo_deim_metrics.csv")

df_detr_speed = pd.read_csv(RESULTS_DIR / "detr_dfine_speed.csv")
df_yolo_speed = pd.read_csv(RESULTS_DIR / "yolo_deim_speed.csv")

df_metrics = pd.concat([df_detr_metrics, df_yolo_metrics], ignore_index=True)
df_speed = pd.concat([df_detr_speed, df_yolo_speed], ignore_index=True)

df = df_metrics.merge(df_speed, on="Модель")

df.to_csv(RESULTS_DIR / "benchmark_all_models.csv", index=False, encoding="utf-8")
