import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def run_evaluation(ann_path, res_path, model_name, plot_save_path):
    cocoGT = COCO(ann_path)
    cocoDT = cocoGT.loadRes(res_path)
    cocoEval = COCOeval(cocoGT, cocoDT, 'bbox')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    metrics = {
        "mAP50-95": round(cocoEval.stats[0], 4),
        "mAP50": round(cocoEval.stats[1], 4),
        "mAP75": round(cocoEval.stats[2], 4),
        "mAP_s": round(cocoEval.stats[3], 4),
        "mAP_m": round(cocoEval.stats[4], 4),
        "mAP_l": round(cocoEval.stats[5], 4),
        "AR1": round(cocoEval.stats[6], 4),
        "AR100": round(cocoEval.stats[8], 4)
    }

    precision = cocoEval.eval['precision']
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
    
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close() 

    return metrics

