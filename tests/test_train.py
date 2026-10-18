"""train.py on small synthetic problems, on CPU."""
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from museum.train import evaluate, fit, predict, train_one_epoch

DEVICE = "cpu"
N_CLASSES = 5
N_FEATURES = 16


def blobs(n, seed, random_labels=False):
    """n points from 5 well-separated Gaussian clusters, as a DataLoader (batches of 16, in order)."""
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % N_CLASSES
    x = 4 * torch.eye(N_CLASSES, N_FEATURES)[y] + torch.randn(n, N_FEATURES, generator=g)
    if random_labels:
        y = torch.randint(0, N_CLASSES, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16)


def mlp(seed=0):
    """Small MLP whose last layer is scaled by 0.01, so the initial loss is ln 5."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(N_FEATURES, 64), nn.ReLU(), nn.Linear(64, N_CLASSES))
    with torch.no_grad():
        model[2].weight.mul_(0.01)
        model[2].bias.zero_()
    return model


class AlwaysClassZero(nn.Module):
    def forward(self, x):
        logits = torch.zeros(len(x), N_CLASSES)
        logits[:, 0] = 1.0
        return logits


def test_evaluate_weights_batches_by_size():
    loader = blobs(103, seed=0)  # last batch has 7 items
    loss, acc = evaluate(AlwaysClassZero(), loader, nn.CrossEntropyLoss(), DEVICE)
    y = loader.dataset.tensors[1]
    p0 = math.e / (math.e + N_CLASSES - 1)
    expected = -(y == 0).float() * math.log(p0) - (y != 0).float() * math.log((1 - p0) / (N_CLASSES - 1))
    assert acc == (y == 0).sum().item() / len(y)
    assert abs(loss - expected.mean().item()) < 1e-6


def test_initial_loss_is_ln5():
    loss, _ = evaluate(mlp(), blobs(200, seed=0), nn.CrossEntropyLoss(), DEVICE)
    assert abs(loss - math.log(N_CLASSES)) < 0.05


def test_overfits_one_batch():
    model = mlp()
    batch = blobs(16, seed=0, random_labels=True)  # random labels: memorisation only
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        loss, acc, _ = train_one_epoch(model, batch, nn.CrossEntropyLoss(), optimizer, DEVICE)
    assert loss < 0.05 and acc == 1.0


def test_fit_learns_and_records_history():
    model = mlp()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    history = fit(model, blobs(400, 0), blobs(100, 1), nn.CrossEntropyLoss(), optimizer, 5, DEVICE, verbose=False)
    per_epoch = ["train_loss", "train_acc", "val_loss", "val_acc", "grad_norms", "epoch_time"]
    assert set(history) == set(per_epoch) | {"best_epoch"}
    assert all(len(history[key]) == 5 for key in per_epoch)
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["val_acc"][-1] > 0.9
    assert 1 <= history["best_epoch"] <= 5 and all(t > 0 for t in history["epoch_time"])
    assert history["grad_norms"] == [{}] * 5


def test_early_stopping_restores_best_weights():
    # Validation labels are random, so the validation loss rises once the model fits the training clusters.
    model = mlp()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    val = blobs(100, 1, random_labels=True)
    history = fit(model, blobs(400, 0), val, loss_fn, optimizer, 100, DEVICE, patience=2, verbose=False)
    best = history["best_epoch"]
    assert len(history["val_loss"]) == best + 2 < 100
    assert history["val_loss"][best - 1] == min(history["val_loss"])
    loss, acc = evaluate(model, val, loss_fn, DEVICE)
    assert abs(loss - history["val_loss"][best - 1]) < 1e-6 and acc == history["val_acc"][best - 1]


def test_grad_norms_cover_trainable_weights_only():
    model = mlp()
    model[0].requires_grad_(False)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    history = fit(model, blobs(64, 0), blobs(32, 1), nn.CrossEntropyLoss(), optimizer, 2, DEVICE,
                  track_grads=True, verbose=False)
    assert [list(norms) for norms in history["grad_norms"]] == [["2.weight"], ["2.weight"]]
    assert all(norms["2.weight"] > 0 for norms in history["grad_norms"])


def test_predict_matches_evaluate():
    model = mlp()
    loader = blobs(103, seed=0)
    y_true, y_pred, probs = predict(model, loader, DEVICE)
    assert y_true.shape == y_pred.shape == (103,) and probs.shape == (103, N_CLASSES)
    assert np.array_equal(y_true, loader.dataset.tensors[1].numpy())
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-6)
    assert np.array_equal(y_pred, probs.argmax(axis=1))
    _, acc = evaluate(model, loader, nn.CrossEntropyLoss(), DEVICE)
    assert abs((y_true == y_pred).mean() - acc) < 1e-12
