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
from realsense_camera import RealSenseCamera
from inference import YOLOInference
from postprocessor import PostProcessor
from publisher import Publisher
from watchdog import Watchdog
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
        self.publisher.publish_status(self.state)

        # Try to initialize all components
        try:
            self.cam = RealSenseCamera()
            self.infer = YOLOInference()
            self.postproc = PostProcessor()
            self.state = "CAPTURING"
        except Exception as e:
            self.get_logger().error(f"Initialization failed: {e}")
            self.state = "ERROR"

        self.publisher.publish_status(self.state)

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

    # ------------------- Core Loop -------------------
    def capture_loop(self):
        """Main capture-inference-postprocess-publish loop"""
        if self.state != "CAPTURING":
            return

        try:
            # Capture frame from camera
            frame = self.cam.capture()

            # Run YOLO inference
            seg_data = self.infer.infer(frame)

            # Postprocess the results
            processed_msg = self.postproc.process(seg_data)

            # Publish segmentation
            self.publisher.publish_segmentation(processed_msg)

        except Exception as e:
            # Any error triggers error state and watchdog intervention
            self.get_logger().error(f"Error during capture loop: {e}")
            self.state = "ERROR"
            self.publisher.publish_status(self.state)

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
