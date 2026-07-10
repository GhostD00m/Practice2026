import os
import csv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from config import DATASET_DIR, ANNOTATIONS, hf_models, PREDICTIONS_DIR, REPORT_DIR

coco = COCO(ANNOTATIONS)
results_dir = PREDICTIONS_DIR
metrics_results = []
target_names = [name.split("/")[-1] for name in hf_models]

for json_path in results_dir.glob("*_results.json"):
    model_name = json_path.stem.replace("_results", "")
    if model_name not in target_names:
        continue
        
    coco_preds = coco.loadRes(str(json_path))
    evaluator = COCOeval(coco, coco_preds, 'bbox')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics_results.append({
        "Модель": model_name,
        "mAP50-95": round(evaluator.stats[0], 4),
        "mAP50": round(evaluator.stats[1], 4),
        "mAP75": round(evaluator.stats[2], 4),
        "mAP small": round(evaluator.stats[3], 4),
        "mAP medium": round(evaluator.stats[4], 4),
        "mAP large": round(evaluator.stats[5], 4),
        "AR1": round(evaluator.stats[6], 4),
        "AR100": round(evaluator.stats[8], 4)
    })

with open(REPORT_DIR / "metrics.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(metrics_results[0].keys()))
    writer.writeheader()
    writer.writerows(metrics_results)
