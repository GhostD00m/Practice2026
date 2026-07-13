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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
    logits = outputs[0][0]
    boxes_norm = outputs[1][0]
    
    scores_matrix = sigmoid(logits)
    results = []

    for i in range(len(logits)):
        class_scores = scores_matrix[i]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]

        if score > conf_thres:
            cx, cy, w, h = boxes_norm[i] * np.concatenate((INPUT_SIZE[:2], INPUT_SIZE[:2]))
            cx = (cx - pad_left) / ratio
            cy = (cy - pad_top)  /ratio
            w = w / ratio
            h = h / ratio

            x1 = cx - w / 2
            y1 = cy - h / 2

            if w > 0 and h > 0:
                results.append({
                    "box": [x1, y1, w, h],
                    "score": float(score),
                    "class_id": int(class_id)
                    })

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
