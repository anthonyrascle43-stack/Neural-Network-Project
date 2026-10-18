"""Models: the feed-forward network (StyleMLP) and the pretrained ResNet."""
from __future__ import annotations

import torch
from torch import nn


class StyleMLP(nn.Module):
    """Feed-forward network: flattened RGB image -> hidden Linear + ReLU (+ Dropout) layers -> class logits.

    Hidden layers use He initialisation, which keeps the activation scale constant through ReLU layers.
    The output layer is scaled by 0.01 so the initial logits are almost equal: the network starts
    with no preference between classes, and the initial loss is ln(n_classes).
    """

    def __init__(self, img_size: int = 64, hidden: tuple[int, ...] = (256, 128), n_classes: int = 5,
                 dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = 3 * img_size * img_size
        for width in hidden:
            layers += [nn.Linear(in_features, width), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = width
        layers.append(nn.Linear(in_features, n_classes))
        self.layers = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linears = [m for m in self.layers if isinstance(m, nn.Linear)]
        for linear in linears:
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
        with torch.no_grad():
            linears[-1].weight.mul_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
