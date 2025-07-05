# stringart_app/tests/test_planner.py

import numpy as np
import pytest

from stringart_app.planner import generate_string_vectors


def test_generate_string_vectors_one_pick_on_nonblank():
    """
    On an image with exactly one dark pixel, greedy should make
    exactly one pick (n_strings=1) without error, and yield
    the proper dict structure.
    """
    img = np.full((30, 30), 255, dtype=np.uint8)
    img[15, 15] = 0  # single pixel to pull string toward

    vecs = generate_string_vectors(
        img,
        n_anchors=8,
        n_strings=1,
        algorithm="greedy"
    )

    # Should have made exactly one pick
    assert isinstance(vecs, list)
    assert len(vecs) == 1

    v = vecs[0]
    assert isinstance(v, dict)
    assert set(v.keys()) == {"from", "to"}
    assert isinstance(v["from"], int)
    assert isinstance(v["to"], int)


def test_generate_string_vectors_blank_raises_value_error():
    """
    On an all-white image, the sampling distribution sums to zero,
    so numpy.choice should raise ValueError.
    """
    blank = np.full((30, 30), 255, dtype=np.uint8)

    with pytest.raises(ValueError) as excinfo:
        generate_string_vectors(
            blank,
            n_anchors=8,
            n_strings=5,
            algorithm="greedy"
        )
    assert "probabilities do not sum to 1" in str(excinfo.value)
