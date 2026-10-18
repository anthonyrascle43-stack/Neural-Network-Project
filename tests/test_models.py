"""StyleMLP: shapes, parameter count, initialisation."""
import math

import pytest
import torch
import torch.nn as nn

from museum.models import StyleMLP, count_parameters


@pytest.mark.parametrize("img_size, hidden", [(32, (256, 128)), (64, (512,)), (16, (64, 32, 16))])
def test_output_shape_and_parameter_count(img_size, hidden):
    model = StyleMLP(img_size=img_size, hidden=hidden)
    assert model(torch.randn(4, 3, img_size, img_size)).shape == (4, 5)
    sizes = [3 * img_size * img_size, *hidden, 5]
    assert count_parameters(model) == sum((n_in + 1) * n_out for n_in, n_out in zip(sizes[:-1], sizes[1:], strict=True))


def test_he_initialisation_and_small_output_layer():
    torch.manual_seed(0)
    model = StyleMLP(img_size=32, hidden=(256, 128))
    linears = [m for m in model.layers if isinstance(m, nn.Linear)]
    assert abs(linears[0].weight.std().item() - math.sqrt(2 / 3072)) < 0.002
    assert abs(linears[-1].weight.std().item() - 0.01 * math.sqrt(2 / 128)) < 0.0005
    assert all(linear.bias.abs().sum() == 0 for linear in linears)


def test_initial_loss_is_ln5():
    torch.manual_seed(0)
    x, y = torch.randn(512, 3, 32, 32), torch.randint(0, 5, (512,))
    loss = nn.CrossEntropyLoss()(StyleMLP(img_size=32)(x), y).item()
    assert abs(loss - math.log(5)) < 0.01


def test_dropout_only_when_asked():
    assert not any(isinstance(m, nn.Dropout) for m in StyleMLP(img_size=16).layers)
    model = StyleMLP(img_size=16, hidden=(64, 32), dropout=0.5)
    assert sum(isinstance(m, nn.Dropout) for m in model.layers) == 2
    model.eval()
    x = torch.randn(2, 3, 16, 16)
    assert torch.equal(model(x), model(x))
