import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from your_msgs.msg import SegmentationOutput, VisionStatus  # replace with actual msg package
from realsense_camera import RealSenseCamera
from inference import YOLOInference
from postprocessor import PostProcessor

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        # Publishers
        self.seg_pub = self.create_publisher(SegmentationOutput, '/vision/segmentation', 10)
        self.status_pub = self.create_publisher(VisionStatus, '/vision/status', 10)

        # Subscriber for logging control
        self.create_subscription(Bool, '/vision/set_logging', self.set_logging_cb, 10)
        self.logging_enabled = False

        # Internal state
        self.state = "INITIALIZING"
        self.publish_state()

        # Initialize components
        try:
            self.cam = RealSenseCamera()
            self.infer = YOLOInference()
            self.postproc = PostProcessor()
            self.state = "CAPTURING"
        except Exception as e:
            self.get_logger().error(f"Initialization failed: {e}")
            self.state = "ERROR"

        self.publish_state()
        # Timer to run capture loop at 10 Hz
        self.timer = self.create_timer(0.1, self.capture_loop)

    def set_logging_cb(self, msg: Bool):
        self.logging_enabled = msg.data
        self.get_logger().info(f"Logging enabled: {self.logging_enabled}")

    def publish_state(self):
        status_msg = VisionStatus()
        status_msg.data = self.state
        self.status_pub.publish(status_msg)

    def capture_loop(self):
        if self.state != "CAPTURING":
            return
        try:
            frame = self.cam.capture()
            seg_data = self.infer.infer(frame)
            processed_msg = self.postproc.process(seg_data)
            self.seg_pub.publish(processed_msg)
        except Exception as e:
            self.get_logger().error(f"Error during capture loop: {e}")
            self.state = "ERROR"
            self.publish_state()


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
