"""Evaluation tools shared by every model: metrics, confidence intervals, McNemar,
CPU latency, model size, confusion matrices and the results/<model>.json files."""
from __future__ import annotations

import io
import json
import math
import pickle
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# Every results/<model>.json has these fields once its family has frozen a candidate (PLAN §3).
REQUIRED_FIELDS = ("model", "config", "val_acc", "n_params", "size_mb", "latency_ms", "train_time_s")
# Added once, in 07_comparison.ipynb, when the test set is scored.
TEST_FIELDS = ("test_acc", "test_ci", "macro_f1")
LATENCY_KEYS = ("threads_1", "threads_2")


# ---------------------------------------------------------------- metrics

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n trials (95 % by default)."""
    assert n > 0 and 0 <= k <= n, f"need 0 <= k <= n and n > 0, got k={k}, n={n}"
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar(y_true, pred_a, pred_b) -> float:
    """Exact (binomial) McNemar test: two-sided p-value that models A and B have the same error rate.

    Only the discordant pairs count: b = A right / B wrong, c = A wrong / B right.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    assert y_true.shape == pred_a.shape == pred_b.shape, "predictions must align with labels"
    right_a, right_b = pred_a == y_true, pred_b == y_true
    b = int(np.sum(right_a & ~right_b))
    c = int(np.sum(~right_a & right_b))
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5).pvalue)


def holm(p_values: dict) -> dict:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error over several tests)."""
    order = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def accuracy_report(y_true, y_pred) -> dict:
    """Test-time fields for save_result: accuracy, its Wilson 95 % CI and macro-F1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k, n = int(np.sum(y_true == y_pred)), len(y_true)
    return {
        "test_acc": accuracy_score(y_true, y_pred),
        "test_ci": list(wilson_ci(k, n)),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


# ---------------------------------------------------------------- cost

def benchmark_latency(predict_one, sample, n_threads: int, warmup: int = 10, runs: int = 100) -> dict:
    """Time predict_one(sample) on CPU with n_threads threads, one call per prediction (batch 1).

    predict_one must cover the whole path (preprocessing + feature extraction + model), and run in
    inference mode for torch models. Both torch and BLAS/OpenMP (numpy, scikit-learn) are limited to
    n_threads, then restored.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(n_threads)
    try:
        with threadpool_limits(limits=n_threads):
            for _ in range(warmup):
                predict_one(sample)
            times = []
            for _ in range(runs):
                start = time.perf_counter()
                predict_one(sample)
                times.append((time.perf_counter() - start) * 1000)
    finally:
        torch.set_num_threads(previous)
    return {"median_ms": statistics.median(times), "p95_ms": float(np.percentile(times, 95))}


def model_size_mb(obj_or_path) -> float:
    """Size in MB (10^6 bytes): of a file or folder on disk, of a torch module's state_dict as
    torch.save writes it, or of any other object (scikit-learn pipeline...) once pickled."""
    if isinstance(obj_or_path, (str, Path)):
        path = Path(obj_or_path)
        assert path.exists(), f"{path} does not exist"
        files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
        return sum(p.stat().st_size for p in files) / 1e6
    buffer = io.BytesIO()
    if isinstance(obj_or_path, torch.nn.Module):
        torch.save(obj_or_path.state_dict(), buffer)
    else:
        pickle.dump(obj_or_path, buffer)
    return buffer.getbuffer().nbytes / 1e6


def total_parameters(model: torch.nn.Module) -> int:
    """Every parameter, frozen ones included (models.count_parameters counts the trainable ones only)."""
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- figures

def plot_confusion(y_true, y_pred, classes, path, title: str | None = None):
    """Confusion matrix: rows = true class, columns = predicted. Shading is the row share (recall
    on the diagonal), each cell shows its count and row %. Saved to path; the figure is returned."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("blue", ["#f7f9fc", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

    fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 2, 1.1 * len(classes) + 1.2))
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ink = "white" if share[i, j] > 0.55 else "#1f2328"
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=9, color=ink)
    labels = [c.replace("_", "-") for c in classes]
    ax.set_xticks(range(len(classes)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), labels)
    ax.set_xlabel("Predicted style")
    ax.set_ylabel("True style")
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"{title + ' · ' if title else ''}accuracy {acc:.1%} (n={cm.sum()})", fontsize=11)
    ax.set_xticks(np.arange(len(classes) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(classes) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------- results files

def _to_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def check_result(result: dict) -> None:
    """Assert that a result record has every required field (and all test fields or none)."""
    missing = [f for f in REQUIRED_FIELDS if f not in result]
    assert not missing, f"{result.get('model', '?')}: missing fields {missing}"
    assert 0 <= result["val_acc"] <= 1, "val_acc must be a fraction, not a percentage"
    for key in LATENCY_KEYS:
        assert {"median_ms", "p95_ms"} <= set(result["latency_ms"].get(key, {})), \
            f"latency_ms needs {LATENCY_KEYS}, each with median_ms and p95_ms"
    present = [f for f in TEST_FIELDS if f in result]
    assert len(present) in (0, len(TEST_FIELDS)), f"partial test fields {present}, need all of {TEST_FIELDS}"
    if present:
        assert 0 <= result["test_acc"] <= 1, "test_acc must be a fraction, not a percentage"


def save_result(name: str, result: dict, results_dir: Path | str = RESULTS_DIR) -> Path:
    """Write results/<name>.json after checking its fields.

    Merges into an existing file, so a family writes its validation part first and
    07_comparison.ipynb adds the test fields (see accuracy_report) later.
    """
    path = Path(results_dir) / f"{name}.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    record.update(result)
    record["model"] = name
    check_result(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=_to_json) + "\n", encoding="utf-8")
    return path


def load_results(results_dir: Path | str = RESULTS_DIR) -> dict[str, dict]:
    """Every results/*.json, checked, keyed by model name."""
    records = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        check_result(record)
        records[record["model"]] = record
    return records
