import os
import time
import json
import cv2
import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from tqdm import tqdm
import sys

MODEL_PATH = sys.argv[1]
OUT_FILE = sys.argv[2]
COCO_ANN_PATH = '../datasets/coco/annotations/instances_val2017.json'
COCO_IMG_PATH = '../datasets/coco/val2017'
INPUT_SIZE = (640, 640)
CONF_THRES = 0.001
IOU_THRES = 0.65

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, left, top

def preprocess(image_path, input_size):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, ratio, pad_left, pad_top = letterbox(img, new_shape=input_size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, ratio, pad_left, pad_top

def postprocess(outputs, ratio, pad_left, pad_top, conf_thres, iou_thres):
    preds = outputs[0][0]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    boxes, scores, class_ids = [], [], []

    for row in preds:
        class_scores = row[4:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]

        if score > conf_thres:
            x1, y1, x2, y2 = row[:4]
            x1 = (x1 - pad_left) / ratio
            y1 = (y1 - pad_top) / ratio
            x2 = (x2 - pad_left) / ratio
            y2 = (y2 - pad_top) / ratio
            w, h = x2 - x1, y2 - y1
            
            if w > 0 and h > 0:
                boxes.append([x1, y1, w, h])
                scores.append(float(score))
                class_ids.append(class_id)

    if len(boxes) == 0:
        return []

    max_wh = 7680 
    boxes_for_nms = [[b[0] + class_ids[i] * max_wh, b[1] + class_ids[i] * max_wh, b[2], b[3]] for i, b in enumerate(boxes)]
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores, conf_thres, iou_thres)

    results = [{"box": boxes[i], "score": scores[i], "class_id": class_ids[i]} for i in indices.flatten()] if len(indices) > 0 else []
    return results

def main():
    coco = COCO(COCO_ANN_PATH)
    img_ids = coco.getImgIds()
    coco_cat_ids = coco.getCatIds()
    class_to_coco_cat = {i: cat_id for i, cat_id in enumerate(coco_cat_ids)}
    
    session = ort.InferenceSession(MODEL_PATH, providers=[sys.argv[3]])
    input_name = session.get_inputs()[0].name

    coco_predictions = []
    times = []
    
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(COCO_IMG_PATH, img_info['file_name'])

        img_tensor, ratio, dw, dh = preprocess(img_path, INPUT_SIZE)

        start_time = time.perf_counter()
        outputs = session.run(None, {input_name: img_tensor})
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)

        detections = postprocess(outputs, ratio, dw, dh, CONF_THRES, IOU_THRES)

        for det in detections:
            coco_predictions.append({
                "image_id": img_id,
                "category_id": class_to_coco_cat[det["class_id"]],
                "bbox": [float(round(x, 3)) for x in det["box"]],
                "score": float(round(det["score"], 5))
            })

    avg_time = sum(times) / len(times)
    print(f"\nСреднее время обработки одного кадра: {avg_time:.2f} ms")

    with open(OUT_FILE, "w") as f:
        json.dump(coco_predictions, f)

if __name__ == "__main__":
    main()
