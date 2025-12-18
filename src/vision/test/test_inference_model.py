import os
import glob
import cv2
import pytest


def test_yolo_inference_on_test_images_real_model():
    """Separate integration test: run actual YOLOInference on images from TEST_IMG_FOLDER.

    This test will be skipped if the model file, test images, or required deps
    are not available. It runs a small number of images on CPU to verify the
    end-to-end inference path (resize + model predict).
    """
    try:
        from vision.inference import YOLOInference
        from vision.config import TEST_IMG_FOLDER, YOLO_MODEL_PATH, YOLO_IMGSZ
    except Exception as e:
        pytest.skip(f"YOLOInference or config import failed: {e}")

    if not os.path.exists(YOLO_MODEL_PATH):
        pytest.skip(f"YOLO model not found at {YOLO_MODEL_PATH}")

    if not os.path.isdir(TEST_IMG_FOLDER):
        pytest.skip(f"Test image folder not found: {TEST_IMG_FOLDER}")

    img_paths = [p for p in glob.glob(os.path.join(TEST_IMG_FOLDER, "*"))
                 if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not img_paths:
        pytest.skip("No test images found in TEST_IMG_FOLDER")

    img_paths = img_paths[:3]

    # Use CPU for portability in CI
    infer = YOLOInference(model_path=YOLO_MODEL_PATH, device="cpu", imgsz=YOLO_IMGSZ)

    for p in img_paths:
        frame = cv2.imread(p)
        assert frame is not None, f"Could not read test image {p}"
        out = infer.infer(frame, overlay=False)
        assert isinstance(out, dict)
        assert set(("masks", "boxes", "scores", "labels", "overlay")).issubset(out.keys())
