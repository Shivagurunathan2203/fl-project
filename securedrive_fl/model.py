from __future__ import annotations

from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mobilenet_v2_tiny(num_classes: int = 10, device: torch.device = None) -> nn.Module:
    """
    MobileNetV2-Tiny backbone -> global average pooling -> softmax classifier.
    num_classes=10 has ~2.23M parameters (2,236,682 params).
    """
    device = device or get_device()
    model = models.mobilenet_v2(weights=None, width_mult=1.0, num_classes=num_classes)
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def flatten_weights(model: nn.Module) -> np.ndarray:
    """
    Flattens model parameters to one float64 vector for HE / GASHE.
    """
    params = [p.detach().cpu().numpy().flatten().astype(np.float64) for p in model.parameters()]
    return np.concatenate(params)


def unflatten_weights(flat_w: np.ndarray, model: nn.Module) -> None:
    """
    Copies flat float64 vector back into model parameters.
    """
    offset = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            chunk = flat_w[offset : offset + numel].reshape(p.shape)
            p.copy_(torch.from_numpy(chunk).to(device))
            offset += numel


def compute_sample_grads(
    model: nn.Module,
    criterion: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> np.ndarray:
    """
    Computes per-sample gradients for batch of size B.
    Returns (B, n_params) array of gradients.
    """
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    b_size = x.shape[0]
    grads = []
    # Per-sample loop
    for i in range(b_size):
        model.zero_grad()
        out = model(x[i : i + 1])
        loss = criterion(out, y[i : i + 1])
        loss.backward()
        g_i = [p.grad.detach().cpu().numpy().flatten().astype(np.float64) for p in model.parameters() if p.grad is not None]
        grads.append(np.concatenate(g_i))
    return np.stack(grads)
