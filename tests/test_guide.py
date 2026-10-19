"""guide.py with a fake predictor and temporary files (CI has no checkpoints or WikiArt images)."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

from museum import guide
from museum.data import CLASSES
from museum.models import StyleMLP
from museum.stories import NOT_SURE


class FixedPredictor:
    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=float)

    def predict_proba(self, images):
        return np.tile(self.probs, (len(images), 1))


def test_answers_with_the_story_when_sure():
    answer = guide.MuseumGuide(FixedPredictor([0.05, 0.05, 0.05, 0.8, 0.05]), threshold=0.5).ask(np.zeros((8, 8, 3)))
    assert answer.style == "Pop Art" and answer.confidence == pytest.approx(0.8)
    assert list(answer.fields) == ["Period", "Story", "What to look for", "Famous example"]
    assert "Warhol" in answer.text


def test_says_not_sure_below_the_threshold():
    answer = guide.MuseumGuide(FixedPredictor([0.3, 0.2, 0.2, 0.2, 0.1]), threshold=0.5).ask(np.zeros((8, 8, 3)))
    assert answer.style is None and list(answer.fields) == ["Message"]
    assert answer.fields == guide.load_stories()[NOT_SURE]


def test_coverage_table_and_threshold():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.55, 0.45]])
    y_true = np.array([0, 1, 1, 0])          # right, wrong, right, right
    table = guide.coverage_table(probs, y_true)
    assert table["threshold"].tolist() == [0.9, 0.8, 0.7, 0.55]
    assert table["coverage"].tolist() == [0.25, 0.5, 0.75, 1.0]
    assert table["accuracy"].tolist() == pytest.approx([1.0, 0.5, 2 / 3, 0.75])
    assert guide.choose_threshold(probs, y_true, target_accuracy=0.75) == 0.55
    assert guide.choose_threshold(probs, y_true, target_accuracy=0.9) == 0.9
    assert guide.choose_threshold(probs, y_true, target_accuracy=1.01) == 1.0   # impossible: never answer


def test_load_photo_crops_the_centre_square(tmp_path):
    photo = np.zeros((100, 200, 3), dtype=np.uint8)
    photo[:, 50:150] = 255                    # the centre square is white, the sides black
    Image.fromarray(photo).save(tmp_path / "photo.png")
    image = guide.load_photo(tmp_path / "photo.png", size=32)
    assert image.shape == (32, 32, 3) and image.dtype == np.uint8 and image.min() > 200


def test_load_predictor_for_the_ffn(tmp_path, monkeypatch):
    config = {"img_size": 8, "hidden": (16,), "dropout": 0.0}
    torch.manual_seed(0)
    model = StyleMLP(**config)
    torch.save({"state_dict": model.state_dict(), "config": config, "mean": (0.5,) * 3, "std": (0.25,) * 3},
               tmp_path / "ffn.pt")
    monkeypatch.setattr(guide, "CHECKPOINTS", tmp_path)
    predictor = guide.load_predictor("ffn")
    probs = predictor.predict_proba(np.zeros((2, 128, 128, 3), dtype=np.uint8))
    assert probs.shape == (2, len(CLASSES)) and np.allclose(probs.sum(axis=1), 1)
    with pytest.raises(ValueError):
        guide.load_predictor("unknown")


def test_render_card(tmp_path):
    answer = guide.MuseumGuide(FixedPredictor([0.8, 0.05, 0.05, 0.05, 0.05]), threshold=0.5).ask(np.zeros((8, 8, 3)))
    fig = guide.render_card(np.zeros((32, 32, 3), dtype=np.uint8), answer, path=tmp_path / "card.png")
    assert (tmp_path / "card.png").stat().st_size > 0
    plt.close(fig)
