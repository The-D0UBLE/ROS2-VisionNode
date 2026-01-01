# Geautomatiseerd testen

Waar staan de tests
- De tests zitten in de map `src/vision/test`.
  - `src/vision/test/test_inference_model.py` — integratie/inferentietest (vereist model + afbeeldingen).
  - `src/vision/test/test_postprocessor.py` — unit-tests voor postprocessing-logica.
  - `src/vision/test/test_realsense_camera.py` — tests voor de camera-wrapper (kan hardware-afhankelijk zijn).

Wat betekenen de tests 
- Integratie-/inference-test (`test_inference_model.py`): controleert dat het model en de inferentie-pijplijn werken samen; vereist vaak het modelbestand en testbeelden.
- Camera-tests: verifiëren dat de camera-wrapper (of mock) correct werkt; kan ge-skip worden als hardware ontbreekt.

Hoe je de tests lokaal runt
- Vanuit de projectroot (waar `pytest.ini` staat) run je alle tests met:

```bash
pytest -q
```

- Alleen één testbestand:

```bash
pytest -q src/vision/test/test_inference_model.py
```

- Alleen tests met een naam/substring:

```bash
pytest -q -k "substring_van_testnaam"
```

- Stop bij de eerste fout:

```bash
pytest -x
```

Tips voor inference-tests
- Integratie-tests hebben vaak een modelbestand nodig, bijvoorbeeld `src/vision/vision/models/best.pt` en testafbeeldingen in `src/vision/testData/Images`.
- Als die bestanden ontbreken, kunnen tests worden overgeslagen (skip) of gemockt.
