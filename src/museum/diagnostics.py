"""Training diagnostics, ported from the Session 4 TP: gradients, activations, initialisation, sanity checks."""
from __future__ import annotations

import torch
from torch import nn

ACTIVATIONS = (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)


def make_mlp(depth: int, activation=nn.ReLU, width: int = 128, in_features: int = 3 * 32 * 32,
             n_classes: int = 5) -> nn.Sequential:
    """Flatten, then `depth` hidden Linear + activation layers of `width` units, then a Linear output (PyTorch's
    default initialisation): the deep networks of the course's gradient and initialisation experiments."""
    layers: list[nn.Module] = [nn.Flatten()]
    for _ in range(depth):
        layers += [nn.Linear(in_features, width), activation()]
        in_features = width
    layers.append(nn.Linear(in_features, n_classes))
    return nn.Sequential(*layers)


def _sequential(model: nn.Module) -> nn.Sequential:
    """The nn.Sequential holding the layers (StyleMLP keeps it in .layers)."""
    return model.layers if hasattr(model, "layers") else model


def layer_grad_norms(model, xb, yb, loss_fn=None) -> list[float]:
    """Gradient norm of each nn.Linear weight on one batch, first layer to last."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.train()
    model.zero_grad(set_to_none=True)
    loss_fn(model(xb), yb).backward()
    return [m.weight.grad.norm().item() for m in _sequential(model) if isinstance(m, nn.Linear)]


@torch.no_grad()
def activation_stds(model, x) -> list[float]:
    """Std of the output of each activation layer, first to last (eval mode, so no dropout)."""
    model.eval()
    stds = []
    for layer in _sequential(model):
        x = layer(x)
        if isinstance(layer, ACTIVATIONS):
            stds.append(x.std().item())
    return stds


@torch.no_grad()
def dead_relu_fraction(model, x) -> list[float]:
    """For each ReLU layer, the share of units that output 0 for every input of x."""
    model.eval()
    fractions = []
    for layer in _sequential(model):
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            fractions.append((x == 0).all(dim=0).float().mean().item())
    return fractions


def init_weights(model, kind: str):
    """Re-initialise every nn.Linear: 'zeros', 'small' N(0, 0.01²), 'large' N(0, 1), 'xavier' or 'he'; biases at 0."""
    for m in _sequential(model):
        if isinstance(m, nn.Linear):
            if kind == "zeros":
                nn.init.zeros_(m.weight)
            elif kind == "small":
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif kind == "large":
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
            elif kind == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif kind == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Unknown initialisation: {kind}")
            nn.init.zeros_(m.bias)
    return model


def overfit_one_batch(model, xb, yb, optimizer, steps: int = 300, loss_fn=None) -> list[float]:
    """Train on a single batch again and again; returns the loss at every step. A working setup drives it to ~0."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    model.train()
    losses = []
    for _ in range(steps):
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
