"""Museum guide demo: photo of a painting -> its style, how sure the model is, and the story of the movement."""
from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from museum.data import CLASSES, IMAGENET_MEAN, IMAGENET_STD, NATIVE_SIZE, ROOT
from museum.models import StyleMLP, TorchPredictor, build_resnet18
from museum.stories import NOT_SURE, load_stories

CHECKPOINTS = ROOT / "checkpoints"
DISPLAY_NAMES = {"Pop_Art": "Pop Art", "Ukiyo_e": "Ukiyo-e"}
# Chosen in notebooks/08_guide.ipynb on the validation set.
GUIDE = {"model": "resnet_frozen", "target_accuracy": 0.9, "threshold": 0.5449433922767639}


def load_predictor(name: str, device: str = "cpu") -> TorchPredictor:
    """Rebuild a candidate from checkpoints/<name>.pt (written by its notebook)."""
    if name != "ffn" and not name.startswith("resnet"):
        raise ValueError(f"No loader for {name!r}")
    checkpoint = torch.load(CHECKPOINTS / f"{name}.pt", map_location="cpu")
    if name == "ffn":
        model = StyleMLP(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return TorchPredictor(model, checkpoint["config"]["img_size"], checkpoint["mean"], checkpoint["std"], device)
    model = build_resnet18(pretrained=False, trainable=checkpoint["trainable"])
    model.load_state_dict(checkpoint["state_dict"])
    return TorchPredictor(model, checkpoint["img_size"], IMAGENET_MEAN, IMAGENET_STD, device)


def load_photo(path: Path, size: int = NATIVE_SIZE) -> np.ndarray:
    """A photo as uint8 [size, size, 3], centre-cropped to a square like the dataset images."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    side = min(rgb.size)
    left, top = (rgb.width - side) // 2, (rgb.height - side) // 2
    square = rgb.crop((left, top, left + side, top + side))
    return np.asarray(square.resize((size, size), Image.Resampling.BILINEAR))


@dataclass
class Answer:
    style: str | None          # None when the guide isn't sure
    confidence: float          # probability of the most likely style
    fields: dict[str, str]     # the story fields, or the "not sure" message

    @property
    def text(self) -> str:
        return "\n\n".join(f"{label}: {value}" for label, value in self.fields.items())


class MuseumGuide:
    """Answers with the style and its story, or with the "not sure" message below the threshold."""

    def __init__(self, predictor, threshold: float, stories: dict | None = None):
        self.predictor = predictor
        self.threshold = threshold
        self.stories = stories or load_stories()

    def ask(self, image: np.ndarray) -> Answer:
        probs = self.predictor.predict_proba(image[None])[0]
        best = int(probs.argmax())
        confidence = float(probs[best])
        if confidence < self.threshold:
            return Answer(None, confidence, self.stories[NOT_SURE])
        style = CLASSES[best]
        return Answer(DISPLAY_NAMES.get(style, style), confidence, self.stories[style])


def coverage_table(probs: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    """For each possible threshold: the share of images the guide answers, and its accuracy on those."""
    confidence = probs.max(axis=1)
    right = probs.argmax(axis=1) == y_true
    order = np.argsort(-confidence)
    answered = np.arange(1, len(order) + 1)
    return pd.DataFrame({"threshold": confidence[order], "coverage": answered / len(order),
                         "accuracy": np.cumsum(right[order]) / answered})


def choose_threshold(probs: np.ndarray, y_true: np.ndarray, target_accuracy: float) -> float:
    """Lowest threshold (so the most answers) whose answered images are at least target_accuracy right."""
    table = coverage_table(probs, y_true)
    ok = table[table["accuracy"] >= target_accuracy]
    return float(ok["threshold"].min()) if len(ok) else 1.0


def render_card(image: np.ndarray, answer: Answer, path=None, width: int = 62):
    """A kiosk-style card: the photo on the left, the guide's answer on the right."""
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1, 1.6]})
    left.imshow(image)
    left.axis("off")
    right.axis("off")
    title = answer.style or "Not sure"
    right.text(0, 1, title, fontsize=20, fontweight="bold", va="top", transform=right.transAxes)
    right.text(0, 0.9, f"confidence {answer.confidence:.0%}", fontsize=10, color="grey", va="top",
               transform=right.transAxes)
    body = "\n\n".join(textwrap.fill(f"{label}: {value}", width) for label, value in answer.fields.items())
    right.text(0, 0.8, body, fontsize=9, va="top", transform=right.transAxes)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("--model", default=GUIDE["model"])
    parser.add_argument("--threshold", type=float, default=GUIDE["threshold"])
    args = parser.parse_args()
    answer = MuseumGuide(load_predictor(args.model), args.threshold).ask(load_photo(args.photo))
    print(f"{answer.style or 'Not sure'} ({answer.confidence:.0%})\n")
    print(answer.text)
