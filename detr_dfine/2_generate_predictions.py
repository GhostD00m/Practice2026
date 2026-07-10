import os
import json
import tqdm
from pycocotools.coco import COCO
from libreyolo import LibreYOLO
from config import DATASET_DIR, ANNOTATIONS, MODELS_ONNX_DIR, hf_models, device, PREDICTIONS_DIR

coco = COCO(ANNOTATIONS)
cat_ids = coco.getCatIds()
img_ids = coco.getImgIds()


target_names = [name.split("/")[-1] for name in hf_models]

def inference(model):
    preds = []
    for img_id in tqdm.tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_path = str(DATASET_DIR / "val2017" / file_name)
        result = model.predict(img_path)
        
        if len(result.boxes) == 0:
            continue
            
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for i in range(len(boxes_xyxy)):
            model_cls_id = int(classes[i])
            if model_cls_id >= len(cat_ids):
                continue
                
            coco_cls_id = cat_ids[model_cls_id]
            x_min, y_min, x_max, y_max = boxes_xyxy[i]
            width = x_max - x_min
            height = y_max - y_min
            
            pred_dict = {
                "image_id": img_id,
                "category_id": coco_cls_id,
                "bbox": [float(x_min), float(y_min), float(width), float(height)],
                "score": float(scores[i])
            }
            preds.append(pred_dict)
            
    return preds

for model_path in MODELS_ONNX_DIR.glob("*.onnx"):
    model_name = model_path.parts[-1]
    model_name = model_name[:model_name.find(".")]
    if model_name not in target_names:
        continue
        
    print(f"\nОбработка {model_name}...")
    output_json = PREDICTIONS_DIR / f"{model_name}_results.json"
    
    if output_json.exists():
        print(f"{model_name} уже обработано")
        continue
        
    model = LibreYOLO(str(model_path), device=device)
    preds = inference(model)
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(preds, f)
        
    print(f"{model_name} обработано")
