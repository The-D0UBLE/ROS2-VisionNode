import sys
import types
import numpy as np


def test_apply_hdr_and_preprocess():
    # Ensure importing the module doesn't fail if pyrealsense2 is absent
    fake_rs = types.ModuleType("pyrealsense2")
    sys.modules["pyrealsense2"] = fake_rs

    from vision.realsense_camera import RealSenseCamera

    # Create instance without calling __init__ to avoid hardware/pipeline setup
    cam = RealSenseCamera.__new__(RealSenseCamera)
    cam.target_size = (64, 64)

    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # _apply_hdr_like should return uint8 array same shape
    out = RealSenseCamera._apply_hdr_like(cam, img)
    assert out.dtype == np.uint8
    assert out.shape == img.shape

    # _preprocess should resize to target_size
    proc = RealSenseCamera._preprocess(cam, img)
    assert proc.shape == (64, 64, 3)
