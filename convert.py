import json
import os
import shutil

INPUT_JSON_DIR = "dataset/Annotations/"
INPUT_IMG_DIR = "dataset/JPEGImages/"

OUTPUT_IMG_DIR = "dataset_yolo/images/"
OUTPUT_LABEL_DIR = "dataset_yolo/labels/"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

ALLOWED_CLASSES = ["lying", "stand"]

YOLO_MAP = {
    "lying" : 0,
    "stand": 1
}

LIMIT = 2000
count = 0

for file in os.listdir(INPUT_JSON_DIR):

    if not file.endswith(".json"):
        continue
    if count >= LIMIT:
        break

    json_path = os.path.join(INPUT_JSON_DIR, file)

    with open(json_path, "r") as f:
        data = json.load(f)

    persons = data["persons"]

    img_name = data["filename"]
    img_path = os.path.join(INPUT_IMG_DIR, img_name)

    if not os.path.exists(img_path):
        continue

    w = data["width"]
    h = data["height"]

    yolo_lines = []

    for p in persons:

        bbox = p["bndbox"]
        actions = p["actions"]

        label_list = [k for k, v in actions.items() if v == 1]

        if len(label_list) == 0:
            continue

        label = label_list[0]

        if label not in ALLOWED_CLASSES:
            continue

        class_id = YOLO_MAP[label]

        xmin = bbox["xmin"]
        ymin = bbox["ymin"]
        xmax = bbox["xmax"]
        ymax = bbox["ymax"]
        x_center = (xmin + xmax) / 2 / w
        y_center = (ymin + ymax) / 2 / h
        width = (xmax - xmin) / w
        height = (ymax - ymin) / h

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    if len(yolo_lines) == 0:
        continue

    shutil.copy(img_path, os.path.join(OUTPUT_IMG_DIR, img_name))

    txt_name = img_name.replace(".jpg", ".txt")
    label_path = os.path.join(OUTPUT_LABEL_DIR, txt_name)

    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))

    count += 1

print(f"Conversão concluída! Total: {count} imagens processadas.")
