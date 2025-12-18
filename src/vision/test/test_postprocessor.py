import sys
import types


def _make_fake_vision_and_std_msgs():
    # Create fake vision_msgs.msg.SegmentationOutput and std_msgs.msg.Header
    vision_msgs = types.ModuleType("vision_msgs")
    vision_msgs.msg = types.ModuleType("vision_msgs.msg")

    class SegmentationOutput:
        def __init__(self):
            self.class_ids = []
            self.scores = []
            self.stamp = None
            self.frame_id = ""

    vision_msgs.msg.SegmentationOutput = SegmentationOutput

    std_msgs = types.ModuleType("std_msgs")
    std_msgs.msg = types.ModuleType("std_msgs.msg")
    class Header:  # placeholder
        def __init__(self):
            self.stamp = None
            self.frame_id = ""

    std_msgs.msg.Header = Header

    return vision_msgs, std_msgs


def test_postprocessor_populates_fields(monkeypatch):
    vision_msgs_mod, std_msgs_mod = _make_fake_vision_and_std_msgs()
    # inject fake modules so `from vision_msgs.msg import SegmentationOutput` works
    sys.modules["vision_msgs"] = vision_msgs_mod
    sys.modules["vision_msgs.msg"] = vision_msgs_mod.msg
    sys.modules["std_msgs"] = std_msgs_mod
    sys.modules["std_msgs.msg"] = std_msgs_mod.msg

    # Provide a fake rclpy.clock.Clock used inside PostProcessor.process
    fake_rclpy = types.ModuleType("rclpy")
    clock_mod = types.ModuleType("rclpy.clock")

    class DummyClock:
        def now(self):
            class T:
                def to_msg(self):
                    return "TIMESTAMP"

            return T()

    # Also provide a rclpy.time module with a Time placeholder so imports succeed
    time_mod = types.ModuleType("rclpy.time")

    class DummyTime:
        def __init__(self):
            pass

    clock_mod.Clock = DummyClock
    time_mod.Time = DummyTime
    sys.modules["rclpy"] = fake_rclpy
    sys.modules["rclpy.clock"] = clock_mod
    sys.modules["rclpy.time"] = time_mod

    # Import the PostProcessor after injecting fakes
    from vision.postprocessor import PostProcessor

    pp = PostProcessor()
    seg = {"labels": [1, 2], "scores": [0.9, 0.5], "masks": [], "boxes": [], "overlay": None}
    msg = pp.process(seg)
    assert list(msg.class_ids) == [1, 2]
    assert list(msg.scores) == [0.9, 0.5]
    # our dummy now().to_msg() returns string "TIMESTAMP"
    assert msg.stamp == "TIMESTAMP"
