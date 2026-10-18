import pickle

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from museum.classical import (
    FEATURES,
    count_learned_values,
    hog_features,
    hsv_histogram,
    make_pipeline,
    raw_pixels,
)


@pytest.fixture
def images():
    return np.random.default_rng(0).integers(0, 256, size=(6, 128, 128, 3), dtype=np.uint8)


def test_feature_shapes(images):
    assert hsv_histogram(images).shape == (6, 12 * 4 * 4)
    assert hog_features(images).shape == (6, 7 * 7 * 2 * 2 * 9)
    assert raw_pixels(images).shape == (6, 32 * 32 * 3)
    for fn in FEATURES.values():
        assert fn(images).dtype == np.float32


def test_hsv_histogram_bins():
    assert np.allclose(hsv_histogram(np.random.default_rng(1).integers(0, 256, (2, 16, 16, 3), np.uint8)).sum(1), 1)
    red = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    red[..., 0] = 255  # hue 0, saturation 1, value 1 -> one single bin
    hist = hsv_histogram(red)[0]
    assert hist.max() == 1.0 and hist.argmax() == (0 * 4 + 3) * 4 + 3


@pytest.mark.parametrize("clf", [LogisticRegression(max_iter=500), SVC(C=1.0)])
def test_pipeline_on_raw_images(images, clf):
    y = np.array([0, 1, 2, 0, 1, 2])
    pipe = make_pipeline("hsv+hog", clf).fit(images, y)
    assert pipe.predict(images).shape == (6,)
    restored = pickle.loads(pickle.dumps(pipe))
    assert (restored.predict(images) == pipe.predict(images)).all()
    assert count_learned_values(pipe) > 0
