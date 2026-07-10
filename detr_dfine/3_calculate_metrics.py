import os
import csv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import matplotlib.pyplot as plt
from config import ANNOTATIONS, hf_models, PREDICTIONS_DIR, RESULTS_DIR

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
    
    precision = evaluator.eval['precision']
    pr_array = precision[:, :, :, 0, 2]
    pr_array = np.where(pr_array > -1, pr_array, np.nan)
    mean_precision_all_iou = np.nanmean(pr_array, axis=(0, 2))
    mean_precision_iou50 = np.nanmean(pr_array[0, :, :], axis=1)
    recalls = np.linspace(0.0, 1.0, 101)

    plt.figure(figsize=(10, 7))
    plt.plot(recalls, mean_precision_all_iou, linewidth=2, label=f'{model_name} mAP@.5:.95')
    plt.plot(recalls, mean_precision_iou50, linewidth=2, linestyle='--', label=f'{model_name} mAP@.50')

    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower left')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    pr_dir = RESULTS_DIR / "pr_curves"
    pr_dir.mkdir(exist_ok=True)
    plt.savefig(pr_dir / f"PR_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

with open(RESULTS_DIR / "detr_dfine_metrics.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(metrics_results[0].keys()))
    writer.writeheader()
    writer.writerows(metrics_results)
