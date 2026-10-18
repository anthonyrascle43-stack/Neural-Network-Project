"""Classical baseline: hand-crafted features + scikit-learn classifiers.

Every feature function maps raw uint8 images [N, H, W, 3] to float32 features [N, D], so a whole
pipeline (features -> scaler -> classifier) runs on the same raw images as the torch models.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import rgb2hsv
from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

HSV_BINS = (12, 4, 4)  # hue gets the most bins: it carries most of the colour information


def _resize(images: np.ndarray, size: int, mode: str = "RGB") -> np.ndarray:
    return np.stack([np.asarray(Image.fromarray(im).convert(mode).resize((size, size), Image.Resampling.BILINEAR))
                     for im in images])


def hsv_histogram(images: np.ndarray, bins: tuple[int, int, int] = HSV_BINS) -> np.ndarray:
    """Joint hue x saturation x value histogram of each image, normalised to sum to 1."""
    hsv = rgb2hsv(images)  # floats in [0, 1], shape [N, H, W, 3]
    idx = np.zeros(hsv.shape[:3], dtype=np.int64)
    for c, n_bins in enumerate(bins):
        idx = idx * n_bins + np.minimum((hsv[..., c] * n_bins).astype(np.int64), n_bins - 1)
    n, size = len(images), int(np.prod(bins))
    flat = idx.reshape(n, -1) + size * np.arange(n)[:, None]
    counts = np.bincount(flat.ravel(), minlength=n * size).reshape(n, size)
    return (counts / counts.sum(axis=1, keepdims=True)).astype(np.float32)


def hog_features(images: np.ndarray, size: int = 64) -> np.ndarray:
    """Histogram of oriented gradients on the grayscale image at size px (edges, strokes, texture)."""
    gray = _resize(images, size, mode="L")
    return np.stack([hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)) for g in gray]
                    ).astype(np.float32)


def colour_hog(images: np.ndarray) -> np.ndarray:
    return np.hstack([hsv_histogram(images), hog_features(images)])


def raw_pixels(images: np.ndarray, size: int = 32) -> np.ndarray:
    """Control: the raw RGB pixels at size px, flattened, in [0, 1]."""
    return (_resize(images, size).reshape(len(images), -1) / 255).astype(np.float32)


FEATURES = {"hsv": hsv_histogram, "hog": hog_features, "hsv+hog": colour_hog, "raw32": raw_pixels}


def make_pipeline(features: str, classifier) -> Pipeline:
    """Raw uint8 images -> features -> standardisation (fitted on train only) -> classifier."""
    return Pipeline([("features", FunctionTransformer(FEATURES[features])),
                     ("scale", StandardScaler()),
                     ("clf", classifier)])


def count_learned_values(pipeline: Pipeline) -> int:
    """Numbers fitted on the training data (the classical counterpart of a parameter count)."""
    scaler, clf = pipeline.named_steps["scale"], pipeline.named_steps["clf"]
    total = scaler.mean_.size + scaler.scale_.size
    if hasattr(clf, "support_vectors_"):
        return total + clf.support_vectors_.size + clf.dual_coef_.size + clf.intercept_.size
    return total + clf.coef_.size + clf.intercept_.size
