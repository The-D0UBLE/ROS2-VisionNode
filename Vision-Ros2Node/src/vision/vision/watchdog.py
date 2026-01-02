# watchdog.py
import time
from threading import Thread

class Watchdog(Thread):
    def __init__(self, vision_node, retry_interval=5.0):
        super().__init__(daemon=True)
        self.vision_node = vision_node
        self.retry_interval = retry_interval
        self.running = True

    def run(self):
        while self.running:
            if self.vision_node.state == "ERROR":
                self.vision_node.get_logger().warn("Watchdog: attempting reinitialization...")
                try:
                    self.vision_node.cam = self.vision_node.cam.__class__()  # Reinit camera
                    self.vision_node.infer = self.vision_node.infer.__class__()  # Reinit YOLO
                    self.vision_node.postproc = self.vision_node.postproc.__class__()  # Reinit postprocessor
                    self.vision_node.state = "CAPTURING"
                    # clear any capturing substates or previous-error state
                    self.vision_node.substate = None
                    self.vision_node.prev_state = None
                    self.vision_node.publish_state()
                    self.vision_node.get_logger().info("Reinitialization successful")
                except Exception as e:
                    self.vision_node.get_logger().error(f"Watchdog reinit failed: {e}")
                    time.sleep(self.retry_interval)
            time.sleep(0.5)  # poll interval
