# ROS2 Vision Node

This repository contains a Python/ROS2 vision module for capturing images, running inference, and publishing results.

Contents
- `src/vision` — main package with nodes, camera wrapper, inference and postprocessing.
- `src/vision/test` — pytest tests.
- `src/vision/vision/models` — (optional) model files such as `best.pt`.

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

2. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Or for development install
pip install -e src/vision
```

3. (Optional) To run the ROS2 node inside a ROS2 workspace, build with `colcon` and source the workspace.

Project structure (brief)
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
- Quick: run all tests with:

```bash
pytest -q
```

Running the ROS2 node (two options)
- Direct (after `pip install -e src/vision`):

```bash
python -m vision.vision_node
```

- As a ROS2 package (build and run via `colcon`):

```bash
# build your workspace with colcon, source the install space
colcon build
source install/setup.bash
ros2 run vision vision_node
```

Getting started for development
- Check `src/vision/vision/config.py` for paths and configuration variables.
- Unit tests (`test_postprocessor.py`) are good examples for quick feedback.

Model evaluation
- For model accuracy and detailed evaluation, create a separate evaluation script or notebook; this is outside unit tests and not covered in this README.

Contributing
- Issues and pull requests welcome. Describe changes briefly and add tests where relevant.

License
- Add your license information here if applicable.
