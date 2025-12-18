import numpy as np

from vision.inference import generate_distinct_colors, draw_legend


def test_generate_distinct_colors_unique():
    n = 5
    colors = generate_distinct_colors(n)
    assert len(colors) == n
    for c in colors:
        assert len(c) == 3
        assert all(0 <= int(x) <= 255 for x in c)
    # ensure at least two colors differ
    assert any(not np.array_equal(colors[0], c) for c in colors[1:])


def test_draw_legend_draws_boxes():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    class_colors = {"a": np.array([10, 20, 30]), "b": np.array([40, 50, 60])}
    out = draw_legend(img, class_colors)
    # legend draws small colored rectangles at (10,10) size (20x20)
    roi = out[10 : 10 + 20, 10 : 10 + 20]
    assert roi.sum() > 0
