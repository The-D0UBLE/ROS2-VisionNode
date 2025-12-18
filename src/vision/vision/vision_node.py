#!/usr/bin/env python3
"""
vision_node.py

- ROS2 node for live RealSense segmentation using YOLO
- Publishes segmentation and node state
- Handles errors via Watchdog
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from vision.realsense_camera import RealSenseCamera
from vision.inference import YOLOInference
from vision.postprocessor import PostProcessor
from vision.publisher import Publisher
from vision.watchdog import Watchdog
from vision_msgs.msg import SegmentationOutput, VisionStatus

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        # Internal logging control
        self.logging_enabled = False
        self.create_subscription(Bool, '/vision/set_logging', self.set_logging_cb, 10)

        # Initialize publishers
        self.publisher = Publisher(self)

        # Initialize internal state
        self.state = "INITIALIZING"
        # `substate` is used when `state` == "CAPTURING" to indicate progress
        self.substate = None
        # `prev_state` stores the previous full state (including substate) when an error occurs
        self.prev_state = None
        self.publish_state()

        # Try to initialize all components
        try:
            self.cam = RealSenseCamera()
            self.infer = YOLOInference()
            self.postproc = PostProcessor()
            self.state = "CAPTURING"
            self.substate = None
        except Exception as e:
            self.get_logger().error(f"Initialization failed: {e}")
            self.state = "ERROR"

        self.publish_state()

        # Start watchdog to handle errors
        self.watchdog = Watchdog(self)
        self.watchdog.start()

        # Timer for capture loop at 10 Hz
        self.timer = self.create_timer(0.1, self.capture_loop)

    # ------------------- Callbacks -------------------
    def set_logging_cb(self, msg: Bool):
        """Enable or disable extra logging in memory"""
        self.logging_enabled = msg.data
        self.get_logger().info(f"Logging enabled: {self.logging_enabled}")

    def publish_state(self):
        # Compose and publish the current status including any substate/prev info
        self.publisher.publish_status(self.state, self.substate, self.prev_state)

    # ------------------- Core Loop -------------------
    def capture_loop(self):
        """Main capture-inference-postprocess-publish loop"""
        if self.state != "CAPTURING":
            return

        try:
            # CAPTURING:CAM
            self.substate = "CAM"
            self.publish_state()
            frame = self.cam.capture()

            # CAPTURING:INFERENCE
            self.substate = "INFERENCE"
            self.publish_state()
            seg_data = self.infer.infer(frame)

            # CAPTURING:POSTPROCESSING
            self.substate = "POSTPROCESSING"
            self.publish_state()
            processed_msg = self.postproc.process(seg_data)

            # done with this cycle, clear substate and publish segmentation
            self.substate = None
            self.publish_state()
            self.publisher.publish_segmentation(processed_msg)

        except Exception as e:
            # Any error triggers error state; capture previous state for debugging
            prev = f"{self.state}"
            if self.substate:
                prev = f"{prev}:{self.substate}"
            self.prev_state = prev
            self.get_logger().error(f"Error during capture loop: {e} (prev={self.prev_state})")
            self.state = "ERROR"
            # Publish immediately and ensure the message is flushed
            try:
                self.publish_state()
                # Process one spin cycle to flush outgoing messages
                rclpy.spin_once(self, timeout_sec=0.01)
            except Exception:
                pass

# ------------------- Entry Point -------------------
def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
