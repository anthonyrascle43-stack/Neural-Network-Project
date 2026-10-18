"""The split and data.py, on random images standing in for WikiArt (which is never committed)."""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import make_split
from museum import data

METADATA = data.ROOT / "data" / "wikiart_style5" / "metadata.csv"


@pytest.fixture(scope="module", autouse=True)
def fake_images(tmp_path_factory):
    """One random 128x128 JPEG per row of split.csv, used in place of IMAGE_DIR."""
    root = tmp_path_factory.mktemp("wikiart_style5")
    rng = np.random.default_rng(0)
    for rel in data.load_split()["path"]:
        (root / rel).parent.mkdir(exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)).save(root / rel)
    patch = pytest.MonkeyPatch()
    patch.setattr(data, "IMAGE_DIR", root)
    data._load_all.cache_clear()
    data.norm_stats.cache_clear()
    yield root
    patch.undo()
    data._load_all.cache_clear()
    data.norm_stats.cache_clear()


def test_split_file_is_reproducible():
    meta = pd.read_csv(METADATA)
    rebuilt = make_split.make_split(meta, n_val=225, n_test=225, seed=0)
    make_split.check(rebuilt, meta, n_val=225, n_test=225)
    pd.testing.assert_frame_equal(rebuilt, data.load_split())


def test_split_sizes_per_class():
    counts = data.load_split().groupby(["split", "label"]).size().unstack()
    assert counts.loc["train"].eq(210).all()
    assert counts.loc["val"].eq(45).all()
    assert counts.loc["test"].eq(45).all()


@pytest.mark.parametrize("size", [32, 128, 224])
def test_get_arrays(size):
    for split, n in [("train", 1050), ("val", 225), ("test", 225)]:
        X, y = data.get_arrays(size, split)
        assert X.shape == (n, size, size, 3) and X.dtype == np.uint8
        assert y.dtype == np.int64 and np.bincount(y).tolist() == [n // 5] * 5
        assert y.flags.writeable


def test_get_arrays_returns_a_copy():
    X, _ = data.get_arrays(32, "val")
    X[:] = 0
    assert data.get_arrays(32, "val")[0].any()


def test_train_normalisation_uses_training_images():
    train_dl, val_dl, test_dl, classes = data.get_loaders(img_size=32, batch_size=256)
    assert classes == data.CLASSES
    assert [len(dl.dataset) for dl in (train_dl, val_dl, test_dl)] == [1050, 225, 225]
    xs = torch.cat([xb for xb, _ in train_dl])
    assert xs.shape[1:] == (3, 32, 32) and xs.dtype == torch.float32
    assert xs.mean(dim=(0, 2, 3)).abs().max() < 1e-3
    assert (xs.std(dim=(0, 2, 3)) - 1).abs().max() < 1e-3


def test_imagenet_normalisation():
    _, val_dl, _, _ = data.get_loaders(img_size=64, normalize="imagenet")
    raw = torch.from_numpy(data.get_arrays(64, "val")[0][0]).permute(2, 0, 1).float() / 255
    mean = torch.tensor(data.IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(data.IMAGENET_STD)[:, None, None]
    assert torch.allclose(val_dl.dataset[0][0], (raw - mean) / std, atol=1e-5)


def test_augmentation_only_on_train():
    train_dl, val_dl, test_dl, _ = data.get_loaders(img_size=64, augment=True)
    torch.manual_seed(1)
    first = train_dl.dataset[0][0]
    torch.manual_seed(2)
    assert first.shape == (3, 64, 64) and not torch.equal(first, train_dl.dataset[0][0])
    for dl in (val_dl, test_dl):
        assert torch.equal(dl.dataset[0][0], dl.dataset[0][0])


def test_shuffle_is_seeded():
    def first_labels(seed):
        return next(iter(data.get_loaders(img_size=32, seed=seed)[0]))[1]

    assert torch.equal(first_labels(0), first_labels(0))
    assert not torch.equal(first_labels(0), first_labels(1))


def test_unknown_normalisation_raises():
    with pytest.raises(ValueError):
        data.get_loaders(normalize="per-image")
