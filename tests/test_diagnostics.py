"""diagnostics.py on small networks: the checks from the Session 4 TP."""
import math

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn

from museum.diagnostics import (
    activation_stds,
    dead_relu_fraction,
    init_weights,
    layer_grad_norms,
    overfit_one_batch,
    plot_history,
)
from museum.models import StyleMLP


def deep_mlp(activation, depth=6, width=64):
    layers = [nn.Flatten()]
    in_features = 3 * 8 * 8
    for _ in range(depth):
        layers += [nn.Linear(in_features, width), activation()]
        in_features = width
    return nn.Sequential(*layers, nn.Linear(in_features, 5))


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    return torch.randn(128, 3, 8, 8, generator=g), torch.randint(0, 5, (128,), generator=g)


def test_grad_norms_vanish_with_deep_sigmoid_not_relu(batch):
    torch.manual_seed(0)
    sigmoid = layer_grad_norms(deep_mlp(nn.Sigmoid), *batch)
    torch.manual_seed(0)
    relu = layer_grad_norms(init_weights(deep_mlp(nn.ReLU), "he"), *batch)
    assert len(sigmoid) == len(relu) == 7
    assert sigmoid[0] / sigmoid[-1] < 1e-2 < relu[0] / relu[-1]


def test_grad_norms_work_on_style_mlp(batch):
    norms = layer_grad_norms(StyleMLP(img_size=8, hidden=(32, 16)), *batch)
    assert len(norms) == 3 and all(n > 0 for n in norms)


def test_activation_stds_by_initialisation(batch):
    x = batch[0]
    stds = {}
    for kind in ["small", "large", "he"]:
        torch.manual_seed(0)
        stds[kind] = activation_stds(init_weights(deep_mlp(nn.ReLU, depth=8, width=256), kind), x)
    assert len(stds["he"]) == 8
    assert stds["small"][-1] < 1e-6 and stds["large"][-1] > 1e6
    assert 0.3 < stds["he"][-1] / stds["he"][0] < 3


def test_dead_relu_fraction(batch):
    torch.manual_seed(0)
    model = init_weights(deep_mlp(nn.ReLU, depth=2), "he")
    assert dead_relu_fraction(model, batch[0])[0] < 0.1
    with torch.no_grad():
        model[1].bias.fill_(-100.0)
    assert dead_relu_fraction(model, batch[0])[0] == 1.0
    assert layer_grad_norms(model, *batch)[0] == 0.0


def test_init_weights_he_and_unknown():
    torch.manual_seed(0)
    layer = init_weights(nn.Sequential(nn.Linear(784, 256)), "he")[0]
    assert abs(layer.weight.std().item() - math.sqrt(2 / 784)) < 0.002 and layer.bias.abs().sum() == 0
    with pytest.raises(ValueError):
        init_weights(nn.Sequential(nn.Linear(4, 4)), "orthogonal")


def test_overfit_one_batch(batch):
    torch.manual_seed(0)
    model = StyleMLP(img_size=8, hidden=(64,))
    xb, yb = batch[0][:16], batch[1][:16]
    losses = overfit_one_batch(model, xb, yb, torch.optim.Adam(model.parameters(), lr=1e-2), steps=200)
    assert len(losses) == 200 and abs(losses[0] - math.log(5)) < 0.1 and losses[-1] < 0.05


def test_plot_history_saves_a_figure(tmp_path):
    history = {"train_loss": [1.6, 1.2], "val_loss": [1.6, 1.4], "train_acc": [0.2, 0.5], "val_acc": [0.2, 0.4]}
    fig = plot_history(history, title="test", path=tmp_path / "curves.png")
    assert (tmp_path / "curves.png").stat().st_size > 0 and len(fig.axes) == 2
    plt.close(fig)
