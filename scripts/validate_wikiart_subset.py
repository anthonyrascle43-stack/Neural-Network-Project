"""Check a generated WikiArt subset against its manifest and metadata."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def validate(root: Path) -> dict:
    info = json.loads((root / "subset_info.json").read_text(encoding="utf-8"))
    with (root / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    classes = sorted(info["classes"])
    class_to_idx = {name: i for i, name in enumerate(classes)}
    counts = Counter(row["label"] for row in rows)
    paths = [row["path"] for row in rows]
    if len(set(paths)) != len(paths) or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate paths or source IDs in metadata.csv")
    if len(rows) != info["total"] or set(counts) != set(classes):
        raise ValueError("Metadata classes or total disagree with subset_info.json")
    for name in classes:
        if counts[name] != info["per_class"] or counts[name] != info["counts"][name]:
            raise ValueError(f"Incomplete or inconsistent class: {name}")

    disk_paths = {p.relative_to(root).as_posix() for name in classes for p in (root / name).glob("*.jpg")}
    if disk_paths != set(paths):
        raise ValueError("Image files and metadata.csv do not contain the same paths")
    folders = {p.name for p in root.iterdir() if p.is_dir()}
    if folders != set(classes):
        raise ValueError("The class folders disagree with subset_info.json")

    for row in rows:
        if int(row["label_idx"]) != class_to_idx[row["label"]]:
            raise ValueError(f"Incorrect ImageFolder class index: {row['path']}")
        if Path(row["path"]).parent.as_posix() != row["label"]:
            raise ValueError(f"Image is outside its labelled class folder: {row['path']}")
        if row[info["label"]] != row["label"]:
            raise ValueError(f"Inconsistent source label: {row['path']}")
        if min(int(row["orig_width"]), int(row["orig_height"])) <= 0:
            raise ValueError(f"Missing original dimensions: {row['path']}")
        with Image.open(root / row["path"]) as image:
            image.load()  # Decode every saved file, detecting truncated or corrupt JPEGs.
            if image.format != "JPEG" or image.mode != "RGB":
                raise ValueError(f"Expected RGB JPEG: {row['path']}")
            if info["crop"] and image.size != (info["size"], info["size"]):
                raise ValueError(f"Incorrect cropped dimensions: {row['path']}")
            if not info["crop"] and min(image.size) != info["size"]:
                raise ValueError(f"Incorrect shorter-side dimension: {row['path']}")

    named_artists = Counter(row["artist"] for row in rows if row["artist"] != "Unknown Artist")
    if info["max_per_artist"] and any(count > info["max_per_artist"] for count in named_artists.values()):
        raise ValueError("The named-artist limit was exceeded")
    return {
        "total": len(rows),
        "counts": {name: counts[name] for name in classes},
        "class_to_idx": class_to_idx,
        "size": info["size"],
        "crop": info["crop"],
        "revision": info["revision"],
        "named_artists": len(named_artists),
        "unknown_artist_images": sum(row["artist"] == "Unknown Artist" for row in rows),
        "all_images_decoded": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/wikiart_style5"))
    args = parser.parse_args()
    print(json.dumps(validate(args.root), indent=2))
