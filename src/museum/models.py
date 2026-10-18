"""Models: the feed-forward network (StyleMLP) and the pretrained ResNet."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from museum.data import make_transform


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


# ---------------------------------------------------------------- pretrained ResNet18

TRAINABLE = {"head": ("fc",), "layer4": ("layer4", "fc"), "all": None}


class StyleResNet(nn.Module):
    """torchvision ResNet18 with a new n_classes head.

    trainable: "head" (frozen features + linear head), "layer4" (fine-tune the last block + head)
    or "all". Frozen blocks stay in eval mode even during model.train(), so their BatchNorm running
    statistics keep their ImageNet values instead of drifting towards our 1,050 training images.
    """

    def __init__(self, n_classes: int = 5, pretrained: bool = True, trainable: str = "layer4"):
        super().__init__()
        assert trainable in TRAINABLE, f"trainable must be one of {list(TRAINABLE)}"
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)
        self.trainable = trainable
        keep = TRAINABLE[trainable]
        for name, param in self.net.named_parameters():
            param.requires_grad = keep is None or name.split(".")[0] in keep

    def frozen_blocks(self) -> list[nn.Module]:
        return [m for m in self.net.children() if not any(p.requires_grad for p in m.parameters())]

    def train(self, mode: bool = True):
        super().train(mode)
        for block in self.frozen_blocks():
            block.eval()
        return self

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The 512-d average-pooled features, just before the head."""
        net = self.net
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
        return torch.flatten(net.avgpool(x), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_resnet18(n_classes: int = 5, pretrained: bool = True, trainable: str = "layer4") -> StyleResNet:
    return StyleResNet(n_classes, pretrained, trainable)


def linear_from_logreg(scaler, logreg) -> nn.Linear:
    """Fold a StandardScaler + multinomial LogisticRegression into one nn.Linear with the same logits,
    so a scikit-learn head can be deployed inside the torch model."""
    weight = logreg.coef_ / scaler.scale_
    bias = logreg.intercept_ - weight @ scaler.mean_
    layer = nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(weight, dtype=torch.float32))
        layer.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
    return layer


class TorchPredictor:
    """Raw images (uint8 [N, H, W, 3], native 128 px) -> predictions, end to end.

    Images are resized to img_size like data.load_image (PIL bilinear), then go through the
    evaluation transform of data.get_loaders. Calling the predictor on one image [H, W, 3] returns
    its class index: that is the batch-1 path timed by evaluation.benchmark_latency.
    """

    def __init__(self, model: nn.Module, img_size: int, mean, std, device: str = "cpu", batch_size: int = 64):
        self.model = model.to(device).eval()
        self.img_size, self.device, self.batch_size = img_size, device, batch_size
        self.transform = make_transform(img_size, tuple(mean), tuple(std), augment=False)

    def to_tensor(self, images: np.ndarray) -> torch.Tensor:
        if images.shape[1:3] != (self.img_size, self.img_size):
            size = (self.img_size, self.img_size)
            images = np.stack([np.asarray(Image.fromarray(im).resize(size, Image.Resampling.BILINEAR))
                               for im in images])
        return self.transform(torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2))

    @torch.inference_mode()
    def _run(self, images: np.ndarray, fn) -> np.ndarray:
        out = [fn(self.to_tensor(images[i:i + self.batch_size]).to(self.device)).cpu()
               for i in range(0, len(images), self.batch_size)]
        return torch.cat(out).numpy()

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        return self._run(images, lambda x: torch.softmax(self.model(x), dim=1))

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.predict_proba(images).argmax(axis=1)

    def embed(self, images: np.ndarray) -> np.ndarray:
        """Penultimate features (model.features), e.g. the 512-d ResNet features."""
        return self._run(images, self.model.features)

    def __call__(self, image: np.ndarray) -> int:
        with torch.inference_mode():
            return int(self.model(self.to_tensor(image[None]).to(self.device)).argmax(dim=1))
