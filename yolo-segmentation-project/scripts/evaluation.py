"""
evaluation.py
--------------
Evaluate a YOLOv8 segmentation model on a folder of images
and display overlayed masks with legends.

Usage:
    python scripts/evaluation.py --model path/to/best.pt --images path/to/test/images
"""

import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO
import numpy as np
import colorsys
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------
# CLI Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Evaluate YOLOv8 segmentation model")
parser.add_argument(
    "--model", type=Path, required=True,
    help="Path to trained YOLOv8 segmentation model (.pt)"
)
parser.add_argument(
    "--images", type=Path, required=True,
    help="Folder containing images to evaluate"
)
parser.add_argument(
    "--data_yaml", type=Path, default=Path("data/yolo-dataset/data.yaml"),
    help="Path to Roboflow data.yaml (contains class names)"
)
parser.add_argument(
    "--imgsz", type=int, default=640, help="Resize images to this size for inference"
)
parser.add_argument(
    "--conf", type=float, default=0.50, help="Confidence threshold for predictions"
)
args = parser.parse_args()

# ----------------------------
# Load class names from data.yaml
# ----------------------------
if not args.data_yaml.exists():
    raise FileNotFoundError(f"{args.data_yaml} not found!")

with open(args.data_yaml) as f:
    data_cfg = yaml.safe_load(f)

CLASS_NAMES = data_cfg.get("names")
if CLASS_NAMES is None or len(CLASS_NAMES) == 0:
    raise ValueError(f"No class names found in {args.data_yaml}")

# ----------------------------
# Load model
# ----------------------------
if not args.model.exists():
    raise FileNotFoundError(f"Model not found: {args.model}")

model = YOLO(args.model)
nc = model.model.nc
if len(CLASS_NAMES) != nc:
    raise ValueError(f"Number of classes in data.yaml ({len(CLASS_NAMES)}) "
                     f"does not match model.nc ({nc})")

print(f"Loaded model from {args.model} with {nc} classes.")

# ----------------------------
# Generate visually distinct colors
# ----------------------------
def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        lightness = 0.5
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        rgb_255 = np.array([int(c*255) for c in rgb])
        colors.append(rgb_255)
    return colors

colors_list = generate_distinct_colors(len(CLASS_NAMES))
CLASS_COLORS = {name: color for name, color in zip(CLASS_NAMES, colors_list)}

# ----------------------------
# Check image folder
# ----------------------------
if not args.images.exists():
    raise FileNotFoundError(f"Image folder not found: {args.images}")

# ----------------------------
# Run evaluation
# ----------------------------
for file in sorted(args.images.glob("*")):
    if not file.is_file():
        continue

    img_pil = Image.open(file).convert('RGB')
    img_resized = img_pil.resize((args.imgsz, args.imgsz))
    img_np = np.array(img_resized)
    overlay = img_np.copy()

    results = model.predict(img_np, imgsz=args.imgsz, conf=args.conf)
    result = results[0]

    legend_handles = []

    for i, mask in enumerate(result.masks.data):
        class_id = int(result.boxes.cls[i].item())
        if class_id >= len(CLASS_NAMES):
            label = f"class{class_id}"
        else:
            label = CLASS_NAMES[class_id]

        color = CLASS_COLORS.get(label, np.array([255,0,0]))
        mask_np = mask.cpu().numpy().astype(bool)
        overlay[mask_np] = (0.6*overlay[mask_np] + 0.4*color).astype(np.uint8)

        if label not in [h.get_label() for h in legend_handles]:
            legend_handles.append(Patch(facecolor=color/255, label=label))

    plt.figure(figsize=(8,8))
    plt.imshow(overlay)
    plt.axis('off')
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05,1), loc='upper left')
    plt.title(file.name)
    plt.show()

