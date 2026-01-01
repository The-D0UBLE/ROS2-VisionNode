# ROS2 Vision Node

Dit repository bevat een Python/ROS2-visionmodule voor het ophalen van beelden, uitvoeren van inferentie en publiceren van resultaten.

Inhoud
- `src/vision` — belangrijkste package met nodes, camera-wrapper, inferentie en postprocessing.
- `src/vision/test` — pytest-tests.
- `src/vision/vision/models` —  modelbestanden bijv `best.pt`.

Belangrijkste features
- Camera-abstractie (RealSense wrapper).
- YOLO-inferentie + postprocessing.
- ROS2-compatible publisher voor segmentatie/vision status messages.

Snelstart

1. Clone de repository:

```bash
git clone <repo-url>
cd ROS2-VisionNode
```

2. Maak een update je Python omgeving en installeer afhankelijkheden:

```bash

pip install -r requirements.txt
```

3. (Optioneel) Als je de ROS2-node wilt draaien binnen een ROS2-workspace, bouw met `colcon` en source je workspace.

Projectstructuur (kort)
- `src/vision/vision/vision_node.py` — hoofd-node / entrypoint.
- `src/vision/vision/realsense_camera.py` — RealSense wrapper.
- `src/vision/vision/inference.py` — inferentielogica.
- `src/vision/vision/postprocessor.py` — postprocessing van modeloutput.
- `src/vision/test` — tests: unit + integratie.

Modellen en testdata
- Plaats modelbestanden in `src/vision/vision/models/` (bijv. `best.pt`).
- Testafbeeldingen worden verwacht in `src/vision/testData/Images` voor integratietests.

Tests
- Zie de uitgebreide testinstructies in [TESTING.md](TESTING.md).

Run de ROS2-node
- Als ROS2-package (bouw en run via `colcon`):

```bash
# bouw je workspace met colcon, source de install space
colcon build
source install/setup.bash
ros2 run vision vision_node
```

Waar te beginnen voor ontwikkeling
- Bekijk `src/vision/vision/config.py` voor paden en configuratievariabelen.