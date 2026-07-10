import os 
import re
import subprocess
import csv
from scripts.evaluate import run_evaluation


MODELS_CONFIG = [
        ("LibreYOLO9", "t", "scripts/inference_LibreYOLO9.py"),
        ("LibreYOLO9", "s", "scripts/inference_LibreYOLO9.py"),
        ("LibreYOLO9", "m", "scripts/inference_LibreYOLO9.py"),
        ("LibreYOLO9", "c", "scripts/inference_LibreYOLO9.py"),
        ("LibreYOLO9E2E", "t", "scripts/inference_LibreYOLO9E2E.py"),
        ("LibreYOLO9E2E", "s", "scripts/inference_LibreYOLO9E2E.py"),
        ("LibreYOLO9E2E", "m", "scripts/inference_LibreYOLO9E2E.py"),
        ("LibreYOLO9E2E", "c", "scripts/inference_LibreYOLO9E2E.py"),
        ("LibreYOLOX", "n", "scripts/inference_LibreYOLOX.py"),
        ("LibreYOLOX", "t", "scripts/inference_LibreYOLOX.py"),
        ("LibreYOLOX", "s", "scripts/inference_LibreYOLOX.py"),
        ("LibreYOLOX", "m", "scripts/inference_LibreYOLOX.py"),
        ("LibreYOLOX", "l", "scripts/inference_LibreYOLOX.py"),
        ("LibreYOLOX", "x", "scripts/inference_LibreYOLOX.py"),
        ("LibreDEIM", "n", "scripts/inference_LibreDEIM.py"),
        ("LibreDEIM", "s", "scripts/inference_LibreDEIM.py"),
        ("LibreDEIM", "m", "scripts/inference_LibreDEIM.py"),
        ("LibreDEIM", "l", "scripts/inference_LibreDEIM.py"),
        ("LibreDEIM", "x", "scripts/inference_LibreDEIM.py"),
        ("LibreDEIMv2", "n", "scripts/inference_LibreDEIMv2.py"),
        ("LibreDEIMv2", "s", "scripts/inference_LibreDEIMv2.py"),
        ("LibreDEIMv2", "m", "scripts/inference_LibreDEIMv2.py"),
        ("LibreDEIMv2", "l", "scripts/inference_LibreDEIMv2.py"),
        ("LibreDEIMv2", "x", "scripts/inference_LibreDEIMv2.py")
        ]

ANN_PATH = "../datasets/coco/annotations/instances_val2017.json"
REPORT_PATH = "Benchmark_Report.md"
PROVIDER = "CPUExecutionProvider"

def main():
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    results_data = []

    for family, size, script_path in MODELS_CONFIG:
        model_name = family + size
        onnx_path = f"weights/{family}/{model_name}.onnx"
        json_path = f"predictions/predictions_{model_name}.json"
        plot_path = f"images/PR_{model_name}.png"

        avg_time = "N/A"

        # Inference
        process = subprocess.Popen(
                ["python3", script_path, onnx_path, json_path, PROVIDER],
                stdout=subprocess.PIPE, text=True
                )
        for line in process.stdout:
            print(line, end="")
            match = re.search(r'Среднее время обработки одного кадра: ([0-9\.]+) ms', line)
            if match:
                avg_time = match.group(1)
        
        process.wait()
        if process.returncode != 0:
            print(f"ERROR Inference {model_name}")
            continue
        print(f"Inference {model_name} end")

        # Metrics
        try:
            metircs = run_evaluation(ANN_PATH, json_path, model_name, plot_path)

            results_data.append({
                "model": model_name,
                "time": avg_time,
                **metircs,
                "plot": plot_path
                })
        except Exception as e:
            print(f"ERROR Metrics {model_name}: {e}")
        print(f"Metrics {model_name} end")
    # Generation .md
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Сравнительный анализ моделей (COCO2017)\n\n")

        f.write("## Сводная таблица метрик\n\n")
        f.write("| Модель | mAP50-95 | mAP50 | mAP75 | mAP small | mAP medium | mAP large | AR1 | AR100 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")

        for row in results_data:
            f.write(f"| **{row['model']}** | {row['mAP50-95']} | {row['mAP50']} | {row['mAP75']} | "
                    f"{row['mAP_s']} | {row['mAP_m']} | {row['mAP_l']} | {row['AR1']} | {row['AR100']} |\n")

        f.write("\n## Графики Precision-Recall\n\n")
        for row in results_data:
            f.write(f"### {row['model']}\n")
            f.write(f"![PR Curve {row['model']}]({row['plot']})\n\n")

    # CSV Generation for final results
    with open("../results/yolo_deim_metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Модель", "mAP50-95", "mAP50", "mAP75", "mAP small", "mAP medium", "mAP large", "AR1", "AR100"])
        for row in results_data:
            writer.writerow([
                row['model'], row['mAP50-95'], row['mAP50'], row['mAP75'], 
                row['mAP_s'], row['mAP_m'], row['mAP_l'], row['AR1'], row['AR100']
            ])

    with open("../results/yolo_deim_speed.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Модель", "FPS", "Time_ms"])
        for row in results_data:
            try:
                time_ms = float(row['time'])
                fps = round(1000 / time_ms, 2)
            except ValueError:
                fps = "N/A"
            writer.writerow([row['model'], fps, row['time']])

if __name__ == "__main__":
    main()
