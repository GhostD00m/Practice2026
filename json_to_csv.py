import os
import json
import csv
import tqdm

PRED_DIR = "predictions"
CSV_DIR = "predictions_csv"

def main():
    os.makedirs(CSV_DIR, exist_ok=True)

    json_files = [f for f in os.listdir(PRED_DIR) if f.endswith('.json')]
    if not json_files:
        print(f"Dir {PRED_DIR} have not got json files")
        return

    headers = ["image_id", "category_id", "score", "x_min", "y_min", "width", "height"]

    for filename in json_files:
        json_path = os.path.join(PRED_DIR, filename)
        csv_filename = filename.replace('.json', '.csv')
        csv_path = os.path.join(CSV_DIR, csv_filename)

        with open(json_path, 'r' , encoding='utf-8') as jf:
            data = json.load(jf)

        with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            writer.writerow(headers)

            for item in data:
                img_id = item.get("image_id")
                cat_id = item.get("category_id")
                score = item.get("score")

                bbox = item.get("bbox", [0, 0, 0, 0])
                x_min, y_min, w, h = bbox

                writer.writerow([img_id, cat_id, score, x_min, y_min, w, h])

if __name__ == "__main__":
    main()