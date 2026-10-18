"""Split, datasets and loaders for the WikiArt style subset."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "data" / "wikiart_style5"
SPLIT_CSV = ROOT / "data" / "split.csv"
CLASSES = ["Baroque", "Cubism", "Impressionism", "Pop_Art", "Ukiyo_e"]
NATIVE_SIZE = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(path: Path = SPLIT_CSV) -> pd.DataFrame:
    """One row per image: path (relative to IMAGE_DIR), label (style name), label_idx, split."""
    split = pd.read_csv(path)
    assert set(split["split"]) == {"train", "val", "test"}, "regenerate it with scripts/make_split.py"
    assert sorted(split["label"].unique()) == CLASSES, "unexpected classes in split.csv"
    assert (split["label"].map(CLASSES.index) == split["label_idx"]).all(), "label_idx must follow CLASSES"
    return split


def load_image(path: Path, img_size: int = NATIVE_SIZE) -> np.ndarray:
    """One RGB image as uint8 [H, W, 3], resized to img_size x img_size if needed."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    if rgb.size != (img_size, img_size):
        rgb = rgb.resize((img_size, img_size), Image.Resampling.BILINEAR)
    return np.asarray(rgb)


@lru_cache(maxsize=4)
def _load_all(img_size: int) -> np.ndarray:
    """Every image listed in split.csv, in file order, as uint8 [N, H, W, 3]."""
    if not (IMAGE_DIR / CLASSES[0]).is_dir():
        raise FileNotFoundError(f"No images in {IMAGE_DIR}: see 'Getting the data' in the README")
    return np.stack([load_image(IMAGE_DIR / rel, img_size) for rel in load_split()["path"]])


def get_arrays(img_size: int = NATIVE_SIZE, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """Images of one split as uint8 [N, H, W, 3] and their labels as int64 [N] (for scikit-learn)."""
    df = load_split()
    mask = (df["split"] == split).to_numpy()
    assert mask.any(), f"unknown split: {split}"
    return _load_all(img_size)[mask], df.loc[mask, "label_idx"].to_numpy(dtype=np.int64, copy=True)


@lru_cache(maxsize=None)
def norm_stats(normalize: str = "train", img_size: int = NATIVE_SIZE) -> tuple[tuple, tuple]:
    """Per-channel (mean, std) in [0, 1]: from the training images only, or the ImageNet values."""
    if normalize == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if normalize != "train":
        raise ValueError(f"normalize must be 'train' or 'imagenet', got {normalize!r}")
    X, _ = get_arrays(img_size, "train")
    mean, std = [], []
    for c in range(3):  # one channel at a time keeps the float copy small
        channel = X[..., c].astype(np.float32) / 255
        mean.append(float(channel.mean()))
        std.append(float(channel.std()))
    return tuple(mean), tuple(std)


def make_transform(img_size: int, mean: tuple, std: tuple, augment: bool = False) -> v2.Compose:
    """uint8 [3, H, W] -> normalised float32 [3, H, W]. No colour jitter: colour is a style cue."""
    steps = []
    if augment:
        steps += [v2.RandomHorizontalFlip(),
                  v2.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True)]
    steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    return v2.Compose(steps)


class StyleDataset(Dataset):
    """Images kept in memory as uint8; each item is (float tensor [3, H, W], label)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: v2.Compose):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.images[i]), self.labels[i]


def get_loaders(img_size: int = NATIVE_SIZE, batch_size: int = 64, augment: bool = False,
                normalize: str = "train", seed: int = 0):
    """(train_dl, val_dl, test_dl, classes). Only the training loader is shuffled and augmented."""
    mean, std = norm_stats(normalize, img_size)
    loaders = []
    for split in ("train", "val", "test"):
        is_train = split == "train"
        X, y = get_arrays(img_size, split)
        dataset = StyleDataset(X, y, make_transform(img_size, mean, std, augment=augment and is_train))
        generator = torch.Generator().manual_seed(seed) if is_train else None
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=is_train, generator=generator))
    return (*loaders, CLASSES)
