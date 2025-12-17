#!/usr/bin/env python3
"""
postprocessor.py

- Converts YOLOInference output into ROS2 SegmentationOutput messages
"""

from vision_msgs.msg.SegmentationOutput import SegmentationOutput
from std_msgs.VisionStatus import Header
import numpy as np
import rospy  # only for header timestamp, optional in ROS2 we can use rclpy.time

class PostProcessor:
    def __init__(self):
        pass

    def process(self, seg_data):
        """
        Convert YOLOInference output to SegmentationOutput message

        seg_data is a dict from YOLOInference.infer():
            masks, boxes, scores, labels, overlay
        """
        msg = SegmentationOutput()

        # Fill in header
        from rclpy.clock import Clock
        from rclpy.time import Time
        clock = Clock()
        msg.header.stamp = clock.now().to_msg()
        msg.header.frame_id = "camera_frame"

        # Fill in arrays
        msg.class_ids = list(seg_data.get("labels", []))
        msg.scores = list(seg_data.get("scores", []))

        # For masks, you could convert boolean arrays to flattened lists
        masks = seg_data.get("masks", [])
        flattened_masks = []
        for mask in masks:
            flattened_masks.append(mask.astype(np.uint8).flatten().tolist())
        msg.masks = flattened_masks

        return msg
