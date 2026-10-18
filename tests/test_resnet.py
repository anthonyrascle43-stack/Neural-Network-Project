import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from museum.data import IMAGENET_MEAN, IMAGENET_STD
from museum.models import TorchPredictor, build_resnet18, linear_from_logreg

# pretrained=False everywhere: CI does not download the ImageNet weights.


@pytest.mark.parametrize("trainable, expected", [("head", 512 * 5 + 5), ("all", None)])
def test_trainable_parameters(trainable, expected):
    model = build_resnet18(pretrained=False, trainable=trainable)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    assert n_trainable == (expected or n_total)


def test_layer4_mode_and_frozen_batchnorm():
    model = build_resnet18(pretrained=False, trainable="layer4").train()
    trainable = {name.split(".")[1] for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"layer4", "fc"}
    assert not model.net.bn1.training and not model.net.layer3.training  # frozen: BN statistics fixed
    assert model.net.layer4.training
    before = model.net.layer1[0].bn1.running_mean.clone()
    model(torch.randn(4, 3, 64, 64))
    assert torch.equal(before, model.net.layer1[0].bn1.running_mean)


def test_predictor_end_to_end():
    torch.manual_seed(0)
    model = build_resnet18(pretrained=False, trainable="head")
    images = np.random.default_rng(0).integers(0, 256, size=(3, 128, 128, 3), dtype=np.uint8)
    for size in (128, 160):  # native size, and upsampled like the 224 px variant
        predictor = TorchPredictor(model, size, IMAGENET_MEAN, IMAGENET_STD)
        assert predictor.to_tensor(images).shape == (3, 3, size, size)
        probs = predictor.predict_proba(images)
        assert probs.shape == (3, 5) and np.allclose(probs.sum(axis=1), 1, atol=1e-5)
        assert predictor(images[0]) == probs[0].argmax()
        assert predictor.embed(images).shape == (3, 512)


def test_linear_from_logreg_matches_sklearn():
    rng = np.random.default_rng(0)
    X, y = rng.normal(2, 3, size=(200, 16)), rng.integers(0, 5, 200)
    scaler = StandardScaler().fit(X)
    logreg = LogisticRegression(max_iter=2000).fit(scaler.transform(X), y)
    layer = linear_from_logreg(scaler, logreg)
    with torch.no_grad():
        probs = torch.softmax(layer(torch.as_tensor(X, dtype=torch.float32)), dim=1).numpy()
    assert np.allclose(probs, logreg.predict_proba(scaler.transform(X)), atol=1e-4)
