"""Write the fixed stratified train/val/test split used by every model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def make_split(meta: pd.DataFrame, n_val: int, n_test: int, seed: int) -> pd.DataFrame:
    # Test first, then validation out of what is left; both stratified on the style.
    rest, test = train_test_split(meta, test_size=n_test, stratify=meta["label"], random_state=seed)
    train, val = train_test_split(rest, test_size=n_val, stratify=rest["label"], random_state=seed)
    split = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    return split[["path", "label", "label_idx", "split"]].sort_values("path").reset_index(drop=True)


def check(split: pd.DataFrame, meta: pd.DataFrame, n_val: int, n_test: int) -> None:
    assert split["path"].is_unique, "an image appears in more than one split"
    assert sorted(split["path"]) == sorted(meta["path"]), "the split and metadata.csv disagree"
    sizes = split["split"].value_counts().to_dict()
    assert sizes == {"train": len(meta) - n_val - n_test, "val": n_val, "test": n_test}, sizes
    per_class = split.groupby(["split", "label"]).size().unstack()
    assert (per_class.nunique(axis=1) == 1).all(), f"unbalanced classes:\n{per_class}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/wikiart_style5/metadata.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/split.csv"))
    parser.add_argument("--val", type=int, default=225, help="validation images")
    parser.add_argument("--test", type=int, default=225, help="test images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata)
    split = make_split(meta, args.val, args.test, args.seed)
    check(split, meta, args.val, args.test)
    split.to_csv(args.out, index=False)
    counts = split.groupby(["split", "label"]).size().unstack().loc[["train", "val", "test"]]
    print(json.dumps({"out": str(args.out), "per_class": counts.to_dict(orient="index")}, indent=2))
