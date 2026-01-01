# ROS2 Vision Node

This repository contains a Python/ROS2 vision module for capturing images, running inference, and publishing results.

Contents
- `src/vision` — main package with nodes, camera wrapper, inference and postprocessing.
- `src/vision/test` — pytest tests.
- `src/vision/vision/models` —  model files such as `best.pt`.

Key features
- Camera abstraction (RealSense wrapper).
- YOLO inference + postprocessing.
- ROS2-compatible publisher for segmentation and vision status messages.

Quick start

1. Clone the repository:

```bash
git clone <repo-url>
cd ROS2-VisionNode
```

2. update ur Python environment and install dependencies:

```bash

pip install -r requirements.txt

```

3.  To run the ROS2 node inside a ROS2 workspace, build with `colcon` and source the workspace.

Project structure
- `src/vision/vision/vision_node.py` — main node / entrypoint.
- `src/vision/vision/realsense_camera.py` — RealSense wrapper.
- `src/vision/vision/inference.py` — inference logic.
- `src/vision/vision/postprocessor.py` — postprocessing of model output.
- `src/vision/test` — tests: unit + integration.

Models and test data
- Place model files in `src/vision/vision/models/` (e.g. `best.pt`).
- Test images are expected in `src/vision/testData/Images` for integration tests.

Tests
- See the detailed testing instructions in [TESTING.md](TESTING.md).

Running the ROS2 node
- As a ROS2 package (build and run via `colcon`):

```bash
# build your workspace with colcon, source the install space
colcon build
source install/setup.bash
ros2 run vision vision_node
```

Getting started for development
- Check `src/vision/vision/config.py` for paths and configuration variables.

