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

    def publish_status(self, status: str, substate: str = None, prev: str = None):
        """Publish a readable status string.

        If `substate` is provided it will be appended as `MAIN:SUB`.
        If `prev` is provided it will be appended to ERROR messages so callers
        can see where the error originated.
        """
        status_msg = VisionStatus()
        if substate:
            composed = f"{status}:{substate}"
        else:
            composed = status

        if status == "ERROR" and prev:
            composed = f"{composed} (prev={prev})"

        status_msg.data = composed
        self.status_pub.publish(status_msg)
