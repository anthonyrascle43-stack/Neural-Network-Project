"""search.py with a fake encoder (CI has neither the CLIP weights nor transformers nor the WikiArt images)."""
import numpy as np
import pandas as pd
import pytest

from museum import search
from museum.data import CLASSES
from museum.guide import DISPLAY_NAMES

NAMES = [DISPLAY_NAMES.get(c, c) for c in CLASSES]


class FakeEncoder:
    """An image filled with the value i is "style i": its embedding is one-hot on i. A text naming a style is
    one-hot on that style, any other text points the same way as every style."""

    def __init__(self):
        self.image_calls = 0

    def encode_images(self, images):
        self.image_calls += 1
        return np.eye(len(CLASSES), dtype=np.float32)[images[:, 0, 0, 0]]

    def encode_texts(self, texts):
        out = np.full((len(texts), len(CLASSES)), 1 / np.sqrt(len(CLASSES)), dtype=np.float32)
        for row, text in enumerate(texts):
            hits = [i for i, name in enumerate(NAMES) if name in text]
            if hits:
                out[row] = np.eye(len(CLASSES))[hits[0]]
        return out


def fake_arrays(img_size, split):
    labels = search.load_split().query("split == @split")["label_idx"].to_numpy()
    return np.broadcast_to(labels[:, None, None, None], (len(labels), 2, 2, 3)).astype(np.uint8), labels


def test_collection_is_train_and_val_with_artist_and_genre():
    table = search.collection_table()
    split = search.load_split()
    assert len(table) == (split["split"] != "test").sum() and set(table["split"]) == {"train", "val"}
    assert table["path"].tolist() == split.query("split == 'train'")["path"].tolist() \
        + split.query("split == 'val'")["path"].tolist(), "rows follow get_arrays order, train then val"
    assert table["artist"].notna().all() and table["genre"].notna().all()


def test_embeddings_are_cached_per_split(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "get_arrays", fake_arrays)
    encoder = FakeEncoder()
    first = search.collection_embeddings(encoder, cache_dir=tmp_path)
    assert encoder.image_calls == 2 and sorted(p.name for p in tmp_path.iterdir()) == ["clip_train.npy", "clip_val.npy"]
    again = search.collection_embeddings(encoder, cache_dir=tmp_path)
    assert encoder.image_calls == 2, "the second call reads the cache"
    np.testing.assert_array_equal(first, again)
    assert first.shape == (len(search.collection_table()), len(CLASSES))


def test_a_cache_of_the_wrong_size_is_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "get_arrays", fake_arrays)
    np.save(tmp_path / "clip_val.npy", np.zeros((3, 5), dtype=np.float32))
    encoder = FakeEncoder()
    embeddings = search.collection_embeddings(encoder, splits=("val",), cache_dir=tmp_path)
    assert encoder.image_calls == 1 and len(embeddings) == len(np.load(tmp_path / "clip_val.npy")) == 225


def test_zero_shot_uses_the_display_names():
    embeddings = np.eye(len(CLASSES), dtype=np.float32)
    scores = search.style_scores(FakeEncoder(), embeddings)
    assert scores.shape == (len(CLASSES), len(CLASSES))
    np.testing.assert_array_equal(scores.argmax(axis=1), np.arange(len(CLASSES)))


def test_search_returns_the_closest_works_first():
    labels = [0, 3, 3, 1, 4, 3]
    table = pd.DataFrame({"path": [f"w{i}.jpg" for i in range(len(labels))], "label": [CLASSES[i] for i in labels]})
    embeddings = np.eye(len(CLASSES), dtype=np.float32)[labels]
    collection = search.Collection(FakeEncoder(), table, embeddings)
    results = collection.search("a Pop Art painting", k=4)
    assert len(results) == 4 and results["score"].is_monotonic_decreasing
    assert results["path"].tolist()[:3] == ["w1.jpg", "w2.jpg", "w5.jpg"], "ties keep the collection order"
    assert results["label"].tolist()[:3] == ["Pop_Art"] * 3 and results["score"].iloc[0] == pytest.approx(1.0)


def test_collection_needs_one_embedding_per_work():
    with pytest.raises(AssertionError):
        search.Collection(FakeEncoder(), pd.DataFrame({"path": ["a", "b"]}), np.zeros((3, 5)))
