from ultralytics import YOLOWorld
from pathlib import Path
from PIL import Image
import numpy as np
import os

# Use environment variable or fallback to local path
model_dir = Path(os.environ.get("YOLO_MODEL_DIR", "data/models/yolo_world_model"))
model_name = "yolov8x-worldv2.pt"
custom_model_name = "custom_yolov8x-worldv2.pt"
image_path = "data/tests/yolo-world/knife-hand-plate-tomato.png"
nouns_file = "data/egtea_gaze/action_annotation/noun_idx.txt"  # format: noun idx

with open(nouns_file, "r") as f:
    nouns = f.readlines()

classes = [noun.split()[0] for noun in nouns]
print(classes)

image = Image.open(image_path)

model = YOLOWorld(str(model_dir / model_name))
model.set_classes(classes)
model.save(str(model_dir / custom_model_name))
del model

model = YOLOWorld(str(model_dir / model_name))
model.set_classes(classes)

custom_model = YOLOWorld(str(model_dir / custom_model_name))


def print_top_k_classes(results, mean_top_k=5):
    top_k_classes = results[0].boxes.cls[:mean_top_k]
    top_k_classes = [classes[int(cls)] for cls in top_k_classes]
    top_k_confidences = [float(conf) for conf in results[0].boxes.conf[:mean_top_k]]
    mean_confidence = np.mean(top_k_confidences)
    print(top_k_classes)
    print(top_k_confidences)
    print(f"Mean confidence: {mean_confidence}")


for i in range(3):
    print(f"Original model {i + 1}")
    results = model.predict(image)
    print_top_k_classes(results)

    print("-" * 40)
    print(f"Custom model {i + 1}")
    custom_results = custom_model.predict(image, half=True, augment=True)
    print_top_k_classes(custom_results)

    print("=" * 100)
