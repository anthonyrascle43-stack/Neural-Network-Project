"""Training diagnostics, ported from the Session 4 TP: gradients, activations, initialisation, sanity checks."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch
from torch import nn

ACTIVATIONS = (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)


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


def plot_history(history: dict, title: str | None = None, path=None, n_classes: int = 5):
    """Train and validation loss and accuracy per epoch, side by side, with chance levels dotted."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for split in ("train", "val"):
        label = "train" if split == "train" else "validation"
        axes[0].plot(epochs, history[f"{split}_loss"], label=label)
        axes[1].plot(epochs, history[f"{split}_acc"], label=label)
    axes[0].axhline(math.log(n_classes), color="grey", ls=":", label=f"ln {n_classes} (chance)")
    axes[1].axhline(1 / n_classes, color="grey", ls=":", label=f"chance ({1 / n_classes:.1f})")
    axes[0].set(title="loss", xlabel="epoch", ylabel="cross-entropy")
    axes[1].set(title="accuracy", xlabel="epoch", ylim=(0, 1.02))
    for ax in axes:
        ax.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
