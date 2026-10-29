"""Text search over the collection with CLIP: "a stormy sea" -> the works that look most like it.

CLIP (Radford et al. 2021) maps images and texts into the same 512-d space, so a text and an image can be compared
with a dot product. We use an open CLIP trained on LAION-2B, with no training on our data at all.
It needs the optional requirements: pip install -r requirements-search.txt. The test set is never searched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from museum.data import CLASSES, IMAGE_DIR, NATIVE_SIZE, ROOT, get_arrays, load_split
from museum.guide import DISPLAY_NAMES

MODEL_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
FEATURES = ROOT / "results" / "features"
COLLECTION = ("train", "val")                 # the test set stays out of everything but 07_comparison
STYLE_PROMPT = "a painting in the {} style"   # the best of three prompts on validation (notebooks/09_search)


class ClipEncoder:
    """Images (uint8 [N, H, W, 3]) and texts -> L2-normalised CLIP embeddings, float32 [N, 512]."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu"):
        from transformers import CLIPModel, CLIPProcessor  # optional dependency, see the module docstring

        self.model = CLIPModel.from_pretrained(model_id).eval().to(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.device = device

    @torch.inference_mode()
    def encode_images(self, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
        out = []
        for start in range(0, len(images), batch_size):
            pixels = self.processor(images=list(images[start:start + batch_size]), return_tensors="pt")
            features = self.model.get_image_features(pixel_values=pixels["pixel_values"].to(self.device))
            out.append(torch.nn.functional.normalize(features.pooler_output, dim=-1).cpu())
        return torch.cat(out).numpy()

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.processor(text=list(texts), return_tensors="pt", padding=True).to(self.device)
        features = self.model.get_text_features(**tokens)
        return torch.nn.functional.normalize(features.pooler_output, dim=-1).cpu().numpy()


def collection_table(splits: tuple[str, ...] = COLLECTION) -> pd.DataFrame:
    """The searchable works, in get_arrays order: path, split, label, label_idx, and the artist and genre
    from metadata.csv ("Unknown Artist" / "Unknown Genre" for most of them)."""
    split = load_split()
    meta = pd.read_csv(IMAGE_DIR / "metadata.csv", usecols=["path", "artist", "genre"])
    parts = [split[split["split"] == name] for name in splits]
    return pd.concat(parts).merge(meta, on="path", how="left", validate="one_to_one").reset_index(drop=True)


def collection_embeddings(encoder, splits: tuple[str, ...] = COLLECTION, cache_dir: Path = FEATURES) -> np.ndarray:
    """CLIP embeddings of the collection images, in collection_table order. Cached per split in results/features/:
    delete clip_*.npy there after changing the model."""
    sizes = load_split()["split"].value_counts()
    parts = []
    for name in splits:
        path = Path(cache_dir) / f"clip_{name}.npy"
        if path.exists() and len(np.load(path, mmap_mode="r")) == sizes[name]:
            parts.append(np.load(path))
            continue
        embeddings = encoder.encode_images(get_arrays(NATIVE_SIZE, name)[0]).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        parts.append(embeddings)
    return np.vstack(parts)


def style_scores(encoder, embeddings: np.ndarray, prompt: str = STYLE_PROMPT) -> np.ndarray:
    """Zero-shot style scores [N, n_classes]: each image's similarity with the prompt for each style.
    The argmax is CLIP's guess, without any training on our images."""
    texts = encoder.encode_texts([prompt.format(DISPLAY_NAMES.get(c, c)) for c in CLASSES])
    return embeddings @ texts.T


class Collection:
    """The searchable works and their embeddings. search() compares a text with every image."""

    def __init__(self, encoder, table: pd.DataFrame, embeddings: np.ndarray):
        assert len(table) == len(embeddings), "one embedding per work"
        self.encoder, self.table, self.embeddings = encoder, table.reset_index(drop=True), embeddings

    @classmethod
    def load(cls, encoder=None, splits: tuple[str, ...] = COLLECTION) -> Collection:
        encoder = encoder or ClipEncoder()
        return cls(encoder, collection_table(splits), collection_embeddings(encoder, splits))

    def search(self, query: str, k: int = 6) -> pd.DataFrame:
        """The k works closest to the query, best first, with their cosine similarity in "score"."""
        scores = self.embeddings @ self.encoder.encode_texts([query])[0]
        top = np.argsort(-scores, kind="stable")[:k]
        return self.table.iloc[top].assign(score=scores[top]).reset_index(drop=True)
