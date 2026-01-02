#!/usr/bin/env python3
"""
inference.py

- Uses YOLO segmentation model on live RealSense camera frames
- Optionally can run on a folder of test images (--seg_test)
- Overlay display for debugging with class legend
"""

import argparse
import os
import cv2
import colorsys
import torch
import numpy as np
from ultralytics import YOLO
from vision.config import YOLO_MODEL_PATH, YOLO_IMGSZ, YOLO_CONF, YOLO_CLASS_NAMES, DEBUG_OVERLAY, TEST_IMG_FOLDER
from vision.realsense_camera import RealSenseCamera


def generate_distinct_colors(n):
    """Generate visually distinct colors using HSV spacing"""
    colors = []
    for i in range(n):
        hue = i / n
        lightness = 0.5
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        rgb_255 = np.array([int(c*255) for c in rgb])
        colors.append(rgb_255)
    return colors


CLASS_COLORS = {name: color for name, color in zip(YOLO_CLASS_NAMES, generate_distinct_colors(len(YOLO_CLASS_NAMES)))}


def draw_legend(img, class_colors):
    """
    Draw a legend on the top-left corner of the image.
    class_colors: dict {class_name: color_array([R,G,B])}
    """
    x, y = 10, 10          # top-left corner of legend
    w, h = 20, 20          # size of color box
    padding = 5
    for class_name, color in class_colors.items():
        cv2.rectangle(img, (x, y), (x + w, y + h), color.tolist(), -1)
        cv2.putText(img, class_name, (x + w + padding, y + h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += h + padding
    return img


class YOLOInference:
    """
    Wrapper for YOLO segmentation inference
    """

    def __init__(self, model_path=YOLO_MODEL_PATH, device=None, imgsz=YOLO_IMGSZ, conf=YOLO_CONF):
        self.model = YOLO(model_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.model.to(self.device)

    def infer(self, frame: np.ndarray, overlay=True):
        """
        Run inference on a single BGR frame.
        Returns:
            dict with keys: masks, boxes, scores, labels, overlay (optional)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=rgb, imgsz=self.imgsz, conf=self.conf, verbose=False)

        if len(results) == 0:
            return {"masks": [], "boxes": [], "scores": [], "labels": [], "overlay": frame if overlay else None}

        r = results[0]

        masks_out, boxes_out, scores_out, labels_out = [], [], [], []
        overlay_img = frame.copy() if overlay else None

        masks_data = getattr(r.masks, "data", None)
        if masks_data is not None:
            masks_list = [m.cpu().numpy().astype(bool) for m in masks_data]

            boxes_np = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else np.zeros((len(masks_list), 4))
            scores_np = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else np.zeros((len(masks_list),))
            classes_np = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, "cls") else np.zeros((len(masks_list),), dtype=int)

            for i, mask in enumerate(masks_list):
                masks_out.append(mask)
                boxes_out.append(boxes_np[i].tolist())
                scores_out.append(float(scores_np[i]))
                labels_out.append(int(classes_np[i]))

                if overlay_img is not None:
                    class_id = labels_out[-1]
                    label = YOLO_CLASS_NAMES[class_id]
                    color = CLASS_COLORS[label]

                    mask_resized = mask
                    if mask_resized.shape[:2] != overlay_img.shape[:2]:
                        mask_resized = cv2.resize(mask_resized.astype("uint8") * 255,
                                                  (overlay_img.shape[1], overlay_img.shape[0]),
                                                  interpolation=cv2.INTER_NEAREST).astype(bool)
                    overlay_img[mask_resized] = (0.6 * overlay_img[mask_resized] + 0.4 * color).astype(np.uint8)

        return {"masks": masks_out, "boxes": boxes_out, "scores": scores_out, "labels": labels_out, "overlay": overlay_img}


# ---------- Main test logic ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seg_test",
        action="store_true",
        help="Run inference on images from a test folder instead of camera",
    )
    args = parser.parse_args()

    infer = YOLOInference()

    if args.seg_test:
        # Loop over all images in the test folder
        for file in os.listdir(TEST_IMG_FOLDER):
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            img_path = os.path.join(TEST_IMG_FOLDER, file)
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[WARN] Could not read {img_path}")
                continue
            out = infer.infer(frame, overlay=True)
            if DEBUG_OVERLAY and out["overlay"] is not None:
                overlay_with_legend = draw_legend(out["overlay"], CLASS_COLORS)
                cv2.imshow("YOLO Overlay", overlay_with_legend)
                cv2.waitKey(0)
    else:
        # Live camera capture
        cam = RealSenseCamera(save_frames=False)
        frame = cam.capture()
        out = infer.infer(frame, overlay=True)
        if DEBUG_OVERLAY and out["overlay"] is not None:
            overlay_with_legend = draw_legend(out["overlay"], CLASS_COLORS)
            cv2.imshow("YOLO Overlay", overlay_with_legend)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
