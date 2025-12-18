#!/usr/bin/env python3
"""
realsense_camera.py

ROS2-ready RealSense Camera wrapper.

Doel:
- Capture RealSense frames
- HDR-like boost + stretch naar 640x640
- Houdt laatste frame in memory
- Optioneel opslaan voor logging
- Klaar voor gebruik in ROS2 nodes
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
from vision.config import CAMERA_TARGET_SIZE, CAMERA_SAVE_FRAMES, CAMERA_OUTPUT_DIR, HDR

class RealSenseCamera:
    def __init__(self, target_size=CAMERA_TARGET_SIZE, save_frames=CAMERA_SAVE_FRAMES, output_dir=CAMERA_OUTPUT_DIR):
        self.target_size = target_size
        self.save_frames = save_frames
        self.output_dir = output_dir
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.last_frame_raw = None
        self.last_frame_processed = None

        if self.save_frames:
            os.makedirs(self.output_dir, exist_ok=True)

    def _apply_hdr_like(self, image: np.ndarray) -> np.ndarray:
        """Simpele HDR-like boost"""
        img = image.astype(np.float32) / 255.0
        img = np.clip((img ** 0.8) * 1.1, 0, 1)
        return (img * 255).astype(np.uint8)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """HDR-like + stretch naar target_size"""
        if HDR:
            img = self._apply_hdr_like(image)
        
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        return img

    def capture(self) -> np.ndarray:
        """
        Capture één frame van de RealSense.
        Update internal state.
        Return processed frame (BGR, 640x640).
        """
        self.pipeline.start(self.config)
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("Geen color frame ontvangen van RealSense")
            frame = np.asanyarray(color_frame.get_data())
            self.last_frame_raw = frame
            processed = self._preprocess(frame)
            self.last_frame_processed = processed

            # Optioneel opslaan
            if self.save_frames:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_path = os.path.join(self.output_dir, f"raw_{timestamp}.jpg")
                proc_path = os.path.join(self.output_dir, f"proc_{timestamp}.jpg")
                cv2.imwrite(raw_path, frame)
                cv2.imwrite(proc_path, processed)

            return processed
        finally:
            self.pipeline.stop()

    def get_last_frame(self) -> np.ndarray:
        """Externe toegang tot laatst gecapte frame (voor inference/logger)."""
        return self.last_frame_processed.copy() if self.last_frame_processed is not None else None


# ---- Test script ----
if __name__ == "__main__":
    cam = RealSenseCamera(save_frames=True)
    frame = cam.capture()
    if frame is not None:
        cv2.imshow("Processed Frame", frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
