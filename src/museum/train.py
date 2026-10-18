"""Training loop, evaluation and prediction, ported from the Session 4 TP."""
from __future__ import annotations

import time

import numpy as np
import torch


def get_device() -> str:
    """Device for training: 'cuda' or 'mps' when available, else 'cpu'. Benchmarks always use the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(model, loader, loss_fn, device) -> tuple[float, float]:
    """(mean loss, accuracy) over the loader, in eval mode."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += loss_fn(logits, yb).item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)
    return total_loss / total, correct / total


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y_true, y_pred, probs) as numpy arrays, in loader order; probs are softmax outputs [N, C]."""
    model.eval()
    y_true, probs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs.append(torch.softmax(model(xb.to(device)), dim=1).cpu())
            y_true.append(yb)
    probs = torch.cat(probs).numpy()
    return torch.cat(y_true).numpy(), probs.argmax(axis=1), probs


def grad_norms(model) -> dict[str, float]:
    """Gradient L2 norm of every trainable weight matrix or kernel (biases skipped)."""
    return {name: p.grad.norm().item() for name, p in model.named_parameters()
            if p.requires_grad and p.ndim > 1 and p.grad is not None}


def train_one_epoch(model, loader, loss_fn, optimizer, device, track_grads=False):
    """One pass in train mode: (mean loss, accuracy, {weight: gradient norm averaged over the batches})."""
    model.train()
    running, correct, total = 0.0, 0, 0
    norms = {}
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        if track_grads:
            for name, value in grad_norms(model).items():
                norms[name] = norms.get(name, 0.0) + value / len(loader)
        optimizer.step()
        running += loss.item() * len(yb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += len(yb)
    return running / total, correct / total, norms


def fit(model, train_loader, val_loader, loss_fn, optimizer, n_epochs, device,
        patience=None, track_grads=False, verbose=True) -> dict:
    """Train for n_epochs and return the history.

    With patience set, training stops once the validation loss has not improved for
    `patience` epochs, and the weights of the best-validation-loss epoch are restored.
    train_acc is measured during the epoch (train mode, augmented data); epoch_time is
    the wall time of training plus validation, in seconds; best_epoch counts from 1.
    """
    model.to(device)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "grad_norms": [], "epoch_time": [], "best_epoch": None}
    best_loss, best_state, waited = float("inf"), None, 0
    for epoch in range(n_epochs):
        start = time.perf_counter()
        train_loss, train_acc, norms = train_one_epoch(model, train_loader, loss_fn, optimizer, device, track_grads)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["grad_norms"].append(norms)
        history["epoch_time"].append(time.perf_counter() - start)
        if verbose:
            print(f"epoch {epoch + 1:2d} | train loss {train_loss:.4f} | train acc {train_acc:.3f} "
                  f"| val loss {val_loss:.4f} | val acc {val_acc:.3f}")

        if val_loss < best_loss:
            best_loss, history["best_epoch"], waited = val_loss, epoch + 1, 0
            if patience is not None:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif patience is not None:
            waited += 1
            if waited >= patience:
                if verbose:
                    print(f"early stop: no val loss improvement for {patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"restored the weights of epoch {history['best_epoch']}")
    return history
