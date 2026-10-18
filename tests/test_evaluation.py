import json

import numpy as np
import pytest
import torch
from scipy.stats import norm

from museum.evaluation import (
    accuracy_report,
    benchmark_latency,
    check_result,
    holm,
    load_results,
    mcnemar,
    model_size_mb,
    plot_confusion,
    save_result,
    total_parameters,
    wilson_ci,
)

Z95 = norm.ppf(0.975)  # reference values below come from statsmodels, which uses the exact quantile


def test_wilson_ci_reference_values():
    assert wilson_ci(81, 100, z=Z95) == pytest.approx((0.7222115, 0.8748525), abs=1e-6)
    low, high = wilson_ci(0, 225, z=Z95)
    assert low == pytest.approx(0, abs=1e-12) and high == pytest.approx(0.0167866, abs=1e-6)
    assert wilson_ci(225, 225)[1] == pytest.approx(1)


def test_mcnemar_exact():
    # 10 pairs where only A is right, 3 where only B is right: statsmodels' exact test gives 0.0923.
    y = np.zeros(100, dtype=int)
    a = np.array([0] * 60 + [1] * 40)
    b = np.array([0] * 50 + [1] * 10 + [0] * 3 + [1] * 37)
    assert mcnemar(y, a, b) == pytest.approx(0.09228515625)
    assert mcnemar(y, a, b) == mcnemar(y, b, a)
    assert mcnemar(y, a, a) == 1.0


def test_holm_matches_statsmodels():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.2})
    assert adjusted == pytest.approx({"a": 0.04, "b": 0.09, "c": 0.09, "d": 0.2})


def test_accuracy_report():
    report = accuracy_report([0, 1, 2, 3], [0, 1, 2, 0])
    assert report["test_acc"] == 0.75
    assert report["test_ci"][0] < 0.75 < report["test_ci"][1]
    assert 0 < report["macro_f1"] < 1


def test_benchmark_latency_restores_threads():
    model = torch.nn.Linear(8, 2).eval()
    before = torch.get_num_threads()
    timing = benchmark_latency(lambda x: model(x), torch.randn(1, 8), n_threads=1, warmup=2, runs=5)
    assert set(timing) == {"median_ms", "p95_ms"} and 0 < timing["median_ms"] <= timing["p95_ms"]
    assert torch.get_num_threads() == before


def test_model_size_and_parameters(tmp_path):
    model = torch.nn.Linear(100, 10)
    assert total_parameters(model) == 1010
    assert 0.004 < model_size_mb(model) < 0.006  # 1,010 float32 values + a little serialisation overhead
    path = tmp_path / "w.pt"
    torch.save(model.state_dict(), path)
    assert model_size_mb(path) == pytest.approx(path.stat().st_size / 1e6)
    assert model_size_mb({"a": list(range(10))}) > 0


def _record(**overrides):
    latency = {"threads_1": {"median_ms": 1.0, "p95_ms": 2.0}, "threads_2": {"median_ms": 1.0, "p95_ms": 2.0}}
    return {"config": {}, "val_acc": np.float64(0.8), "n_params": 10, "size_mb": 0.1,
            "latency_ms": latency, "train_time_s": 1.0, **overrides}


def test_save_result_merges_and_checks(tmp_path):
    save_result("demo", _record(), tmp_path)
    path = save_result("demo", accuracy_report([0, 1], [0, 1]), tmp_path)
    record = json.loads(path.read_text())
    assert record["model"] == "demo" and record["test_acc"] == 1.0 and record["val_acc"] == 0.8
    assert list(load_results(tmp_path)) == ["demo"]


@pytest.mark.parametrize("bad", [
    _record(val_acc=80),                                   # a percentage
    {k: v for k, v in _record().items() if k != "size_mb"},  # missing field
    _record(test_acc=0.8),                                 # partial test fields
    _record(latency_ms={"threads_1": {"median_ms": 1.0}}),  # incomplete latency
])
def test_check_result_rejects(bad):
    with pytest.raises(AssertionError):
        check_result({"model": "bad", **bad})


def test_plot_confusion_writes_file(tmp_path):
    path = tmp_path / "figures" / "cm.png"
    plot_confusion([0, 1, 2, 2], [0, 2, 2, 2], ["a", "b", "c"], path, title="demo")
    assert path.stat().st_size > 0
