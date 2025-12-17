import os

# ---------- Camera ----------
CAMERA_TARGET_SIZE = (640, 640)
CAMERA_SAVE_FRAMES = False
CAMERA_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")

# ---------- YOLO ----------
# Source path (development)
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # vision/
MODEL_PATH_SRC = os.path.join(SRC_DIR, "vision/models", "best.pt")

# Installed path (after colcon build)
INSTALL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # fallback
MODEL_PATH_INSTALL = os.path.join(INSTALL_DIR, "vision/models", "best.pt")

YOLO_MODEL_PATH = MODEL_PATH_SRC if os.path.exists(MODEL_PATH_SRC) else MODEL_PATH_INSTALL
YOLO_IMGSZ = 640
YOLO_CONF = 0.5
YOLO_CLASS_NAMES = ['boat', 'dock', 'grass', 'reed', 'sky', 'tree', 'wall', 'water']

# ---------- Test ----------
TEST_IMG_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testData", "Images")


# ---------- Misc ----------
DEBUG_OVERLAY = True

