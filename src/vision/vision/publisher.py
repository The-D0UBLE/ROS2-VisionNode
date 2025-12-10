# publisher.py
import rclpy
from rclpy.node import Node
from vision_msgs.msg import SegmentationOutput, VisionStatus  

class Publisher:
    def __init__(self, node: Node):
        self.node = node
        self.seg_pub = node.create_publisher(SegmentationOutput, '/vision/segmentation', 10)
        self.status_pub = node.create_publisher(VisionStatus, '/vision/status', 10)

    def publish_segmentation(self, seg_msg: SegmentationOutput):
        self.seg_pub.publish(seg_msg)

    def publish_status(self, status: str):
        status_msg = VisionStatus()
        status_msg.data = status
        self.status_pub.publish(status_msg)
