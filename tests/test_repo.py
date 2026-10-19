"""Project rules from PLAN.md that need no images: metadata, committed files, notebooks."""
import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
TEST_SET_NOTEBOOK = "07_comparison.ipynb"
# A call that trains, scores, predicts or tunes with test data: evaluate(model, test_dl, ...), clf.score(X_test, ...),
# fit(model, train_dl, test_dl, ...), coverage_table(probs, y_test)... A tripwire, not a proof.
TEST_SET_USE = re.compile(
    r"\b(evaluate|predict\w*|score|\w+_score|fit|coverage_table|choose_threshold|ask)\s*\(.*(?<![A-Za-z])test")


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ROOT / line for line in out.splitlines()]


def code_cells(path: Path) -> list[dict]:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return [c for c in cells if c["cell_type"] == "code" and "".join(c["source"]).strip()]


def test_metadata_matches_manifest():
    info = json.loads((ROOT / "data/wikiart_style5/subset_info.json").read_text(encoding="utf-8"))
    meta = pd.read_csv(ROOT / "data/wikiart_style5/metadata.csv")
    classes = sorted(info["classes"])
    assert len(meta) == info["total"] and meta["path"].is_unique and meta["id"].is_unique
    assert meta["label"].value_counts().to_dict() == info["counts"]
    assert (meta["label"].map(classes.index) == meta["label_idx"]).all(), "class indices must be alphabetical"
    assert (meta["path"].str.split("/").str[0] == meta["label"]).all(), "every image sits in its class folder"


def test_no_images_or_checkpoints_committed():
    """WikiArt images stay local (licence unknown); weights and cached features are rebuilt by the notebooks."""
    files = [p for p in tracked_files() if p.exists()]
    banned = [p for p in files if p.suffix.lower() in {".jpg", ".jpeg", ".zip", ".pt", ".pth", ".ckpt", ".npy", ".npz"}]
    assert not banned, f"remove from git: {banned}"
    figure_dirs = (ROOT / "results/figures", ROOT / "slides")
    stray = [p for p in files
             if p.suffix.lower() in {".png", ".webp"} and not any(p.is_relative_to(d) for d in figure_dirs)]
    assert not stray, f"figures belong in results/figures/: {stray}"
    large = [p for p in files if p.stat().st_size > 10_000_000 and not p.is_relative_to(ROOT / "slides")]
    assert not large, f"files over 10 MB: {large}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_ran_top_to_bottom(path):
    """Notebook outputs are our evidence: every cell ran once, in order, without an error left behind."""
    cells = code_cells(path)
    counts = [c["execution_count"] for c in cells]
    assert counts == list(range(1, len(counts) + 1)), "cells weren't run once, in order: Restart & Run All, then commit"
    errors = [o["ename"] for c in cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, f"error outputs: {errors}"


@pytest.mark.parametrize("path", [p for p in NOTEBOOKS if p.name != TEST_SET_NOTEBOOK], ids=lambda p: p.name)
def test_test_set_only_scored_in_comparison(path):
    """PLAN.md §3: the four candidates are scored on the test set once, together, in 07_comparison."""
    hits = [line.strip() for c in code_cells(path) for line in "".join(c["source"]).splitlines()
            if TEST_SET_USE.search(line)]
    assert not hits, f"scores on test data outside {TEST_SET_NOTEBOOK}: {hits}"


@pytest.mark.parametrize("line, uses_test", [
    ("evaluate(model, test_dl, loss_fn, device)", True),
    ("y_true, y_pred, probs = predict(model, test_dl, device)", True),
    ("clf.score(X_test, y_test)", True),
    ("accuracy_score(y_test, clf.predict(X_test))", True),
    ("clf.predict_proba(X_test)", True),
    ("evaluate(model.to(device), test_dl, loss_fn, device)", True),
    ("fit(model, train_dl, test_dl, loss_fn, optimizer, 5, device)", True),
    ("threshold = choose_threshold(probs, y_test, 0.9)", True),
    ("guide.ask(X_test[0])", True),
    ("fit(model, train_dl, val_dl, loss_fn, optimizer, 5, device)", False),
    ("clf.fit(X_train, y_train)", False),
    ("evaluate(model, val_dl, loss_fn, device)", False),
    ("clf.predict(X_val)", False),
    ("train_dl, val_dl, _, classes = get_loaders()", False),
    ("predict(model, latest_dl, device)", False),
])
def test_test_set_pattern(line, uses_test):
    assert bool(TEST_SET_USE.search(line)) == uses_test
