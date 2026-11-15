from __future__ import annotations

import math
from typing import Optional, Callable
import numpy as np

from securedrive_fl.config import Config


# ═════════════════════════════════════════════════════════════════════════════
# 2. TOY MODEL  (softmax regression stands in for V56's 1280-256-10 MLP;
#                per-sample gradients are closed-form so DP-SGD needs no autodiff)
# ═════════════════════════════════════════════════════════════════════════════
def n_params(d: int, K: int) -> int:
    return d * K + K


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _forward(w: np.ndarray, X: np.ndarray, K: int) -> np.ndarray:
    d = X.shape[1]
    return _softmax(X @ w[: d * K].reshape(d, K) + w[d * K:])


def per_sample_grads(w: np.ndarray, X: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    """(B, n_params) gradient of cross-entropy for every sample."""
    P = _forward(w, X, K)
    P[np.arange(len(y)), y] -= 1.0                       # dL/dlogits = p - onehot
    gW = X[:, :, None] * P[:, None, :]                   # (B, d, K)
    return np.concatenate([gW.reshape(len(y), -1), P], axis=1)


def evaluate(w: np.ndarray, X: np.ndarray, y: np.ndarray, K: int) -> tuple[float, float]:
    P = _forward(w, X, K)
    acc = float((P.argmax(1) == y).mean())
    loss = float(-np.log(P[np.arange(len(y)), y] + 1e-9).mean())
    return acc, loss


def predict(w: np.ndarray, X: np.ndarray, K: int) -> np.ndarray:
    P = _forward(w, X, K)
    return P.argmax(axis=1)


def evaluate_full(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    target_class: int = 2,
    patch_size: int = 4,
    trigger_val: float = 0.6,
) -> dict[str, float]:
    """
    Computes complete evaluation metrics on the test set:
    accuracy, loss, Attack Success Rate (ASR), macro-F1, and macro-precision.
    """
    from securedrive_fl.attacks import stamp_backdoor_trigger
    from securedrive_fl.metrics import compute_metrics

    P = _forward(w, X, K)
    acc = float((P.argmax(1) == y).mean())
    loss = float(-np.log(P[np.arange(len(y)), y] + 1e-9).mean())

    y_pred = P.argmax(axis=1)
    X_tr = stamp_backdoor_trigger(X, patch_size=patch_size, trigger_val=trigger_val)
    P_tr = _forward(w, X_tr, K)
    y_pred_backdoor = P_tr.argmax(axis=1)

    m = compute_metrics(
        y_true=y,
        y_pred=y_pred,
        target_class=target_class,
        y_pred_backdoor=y_pred_backdoor,
    )
    return {
        "acc": acc,
        "loss": loss,
        "asr": m["asr"],
        "macro_f1": m["macro_f1"],
        "macro_precision": m["macro_precision"],
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3. CLIENT-SIDE LOCAL TRAINING  (honest DP-SGD, or backdoor/ascent poisoner)
# ═════════════════════════════════════════════════════════════════════════════
def _local_sgd_loop(
    w_global: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    cfg: Config,
    rng: np.random.Generator,
    sign: float = 1.0,
    grad_fn: Optional[Callable] = None,
) -> np.ndarray:
    if grad_fn is None:
        grad_fn = lambda w_curr, x_b, y_b: per_sample_grads(w_curr, x_b, y_b, K)

    w = w_global.copy()
    is_fedprox = (cfg.aggregator == "fedprox")

    for _ in range(cfg.local_epochs):
        idx = rng.permutation(len(X))
        for s in range(0, len(X) - cfg.batch + 1, cfg.batch):
            b = idx[s: s + cfg.batch]
            G = grad_fn(w, X[b], y[b])
            if cfg.use_dp:
                norms = np.linalg.norm(G, axis=1, keepdims=True)
                G = G * np.minimum(1.0, cfg.clip_C / (norms + 1e-9))
                g = (G.sum(0) + rng.normal(0.0, cfg.sigma, G.shape[1])) / cfg.batch
            else:
                g = G.mean(0)

            if is_fedprox and sign > 0:
                g = g + cfg.prox_mu * (w - w_global)

            w -= cfg.lr * sign * g

    return w - w_global


def local_train(
    w_global: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    cfg: Config,
    rng: np.random.Generator,
    ascent: bool = False,
    poison: bool = False,
    grad_fn: Optional[Callable] = None,
) -> np.ndarray:
    """
    Returns Δw = w_local - w_global.

    DP-SGD per mini-batch (Abadi et al. 2016):
        g_i  = per-sample gradient
        ĝ_i  = g_i * min(1, C / ||g_i||)                  clip
        g    = (Σ ĝ_i + N(0, sigma² I)) / B               noise, sigma = z*C

    Under poisoning:
      - backdoor mode (Paper Eq. 13):
          Δw_m = alpha * Δw_target + (1 - alpha) * Δw_benign
      - ascent mode (hld fallback):
          sign = -1.0
    """
    is_poisoned = poison or (cfg.attack == "model_poison" and ascent)

    # 1. Backdoor Poisoning per Paper Eq. 13
    if is_poisoned and getattr(cfg, "poison_mode", "backdoor") == "backdoor" and not ascent:
        from securedrive_fl.attacks import stamp_backdoor_trigger
        X_tr = stamp_backdoor_trigger(X)
        target_c = getattr(cfg, "target_class", 2)
        y_tr = np.full_like(y, target_c)

        alpha = getattr(cfg, "poison_alpha", 0.5)
        dw_target = _local_sgd_loop(w_global, X_tr, y_tr, K, cfg, rng, sign=1.0, grad_fn=grad_fn)
        dw_benign = _local_sgd_loop(w_global, X, y, K, cfg, rng, sign=1.0, grad_fn=grad_fn)

        dw = alpha * dw_target + (1.0 - alpha) * dw_benign
        if not cfg.use_dp:
            boost = getattr(cfg, "poison_boost", 1.0)
            dw = dw * boost
            if getattr(cfg, "norm_clip", False):
                norm_dw = np.linalg.norm(dw)
                if norm_dw > cfg.clip_C:
                    dw = dw * (cfg.clip_C / (norm_dw + 1e-9))
        return dw

    # 2. Honest training or gradient ascent
    sign = -1.0 if (ascent or (is_poisoned and getattr(cfg, "poison_mode", "backdoor") == "ascent")) else 1.0
    dw = _local_sgd_loop(w_global, X, y, K, cfg, rng, sign=sign, grad_fn=grad_fn)
    if not cfg.use_dp:
        if is_poisoned:
            boost = getattr(cfg, "poison_boost", 1.0)
            dw = dw * boost
        if getattr(cfg, "norm_clip", False):
            norm_dw = np.linalg.norm(dw)
            if norm_dw > cfg.clip_C:
                dw = dw * (cfg.clip_C / (norm_dw + 1e-9))
    return dw
