# Automated Testing

## Test Locations

* Tests are located in the `src/vision/test` directory:

  * `src/vision/test/test_inference_model.py` — integration/inference test (requires model + images).
  * `src/vision/test/test_postprocessor.py` — unit tests for postprocessing logic.
  * `src/vision/test/test_realsense_camera.py` — tests for the camera wrapper (may be hardware-dependent).

## Test Descriptions

* **Integration / Inference Test (`test_inference_model.py`)**: verifies that the model and inference pipeline work together; often requires the model file and test images.
* **Camera Tests**: check that the camera wrapper (or mock) works correctly; can be skipped if hardware is unavailable.

## Running Tests Locally

* From the project root (where `pytest.ini` is located), run all tests with:

```bash
pytest -q
```

* Run only a single test file:

```bash
pytest -q src/vision/test/test_inference_model.py
```

* Run tests matching a name or substring:

```bash
pytest -q -k "substring_of_test_name"
```

* Stop at the first failure:

```bash
pytest -x
```

## Tips for Inference Tests

* Integration tests often require a model file, e.g., `src/vision/vision/models/best.pt`, and test images in `src/vision/testData/Images`.
* If these files are missing, tests can be skipped or mocked.

