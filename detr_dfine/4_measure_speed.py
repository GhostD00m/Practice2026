import os
import time
import csv
from pycocotools.coco import COCO
from libreyolo import LibreYOLO
from config import DATASET_DIR, ANNOTATIONS, MODELS_ONNX_DIR, hf_models, device, REPORT_DIR

coco = COCO(ANNOTATIONS)
img_ids = coco.getImgIds()
speed_results = []
target_names = [name.split("/")[-1] for name in hf_models]

for model_path in MODELS_ONNX_DIR.rglob('*.onnx'):
    model_name = model_path.parts[-1]
    model_name = model_name[:model_name.find('.')]
    if model_name not in target_names:
        continue
        
    print(f'\nОбработка {model_name}...')
    model = LibreYOLO(str(model_path), device=device)

    test_paths = []
    for img_id in img_ids[:220]:
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_path = str(DATASET_DIR / 'val2017' / file_name)
        test_paths.append(img_path)
    
    for img_warmup in test_paths[:20]:
        _ = model.predict(img_warmup)

    start = time.perf_counter()
    for img in test_paths[20:]:    
        _ = model.predict(img)
    end = time.perf_counter()

    res = (end - start) / 200
    fps = 1 / res
    
    speed_results.append({
        'Модель': model_name,
        'FPS': round(fps, 2),
        'Time_ms': round(res * 1000, 2)
    })

with open(REPORT_DIR / "speed.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=['Модель', 'FPS', 'Time_ms'])
    writer.writeheader()
    writer.writerows(speed_results)
