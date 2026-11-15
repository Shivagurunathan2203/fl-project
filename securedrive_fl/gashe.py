from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

from securedrive_fl.config import Config
from securedrive_fl.he import HEBackend


def gashe_mask(dw: np.ndarray, cfg: Config) -> tuple[np.ndarray, float, str]:
    """Return (M as bool array, tau, which rule fired)."""
    a = np.abs(dw)
    n = len(dw)
    if cfg.use_dp:                                       # SecureDrive-FL
        tau, tier = cfg.sigma, "primary"                 # tau = sigma_noise = z*C
        M = a >= tau
        if not M.any():                                  # fallback 1: per-parameter clip share
            tau, tier = cfg.clip_C / (math.sqrt(n) + 1e-8), "dp_floor"
            M = a >= tau
            if not M.any():                              # fallback 2: last resort
                M = np.zeros(n, dtype=bool)
                M[np.argmax(a)] = True
                tier = "argmax"
        return M, tau, tier
    k = max(1, int(round(cfg.top_k * n)))                # GASHE-only: rank-based top-k
    M = np.zeros(n, dtype=bool)
    M[np.argpartition(-a, k - 1)[:k]] = True
    return M, float(np.partition(a, n - k)[n - k]) if k < n else 0.0, "topk"


@dataclass
class Payload:
    cid: int
    n_examples: int
    plain: np.ndarray                  # GASHE: p = (1-M)⊙Δw   |  plain path: full Δw
    ciphertext: Any = None             # GASHE: Enc(M⊙Δw)      |  plain path: None
    mask: Optional[np.ndarray] = None  # only read by the leakage metric, not by aggregation
    tier: str = "n/a"
    scale: Optional[float] = None


def make_payload(cid: int, dw: np.ndarray, n_examples: int, cfg: Config, he: HEBackend) -> Payload:
    if getattr(cfg, "batchcrypt", False):
        from securedrive_fl.baselines import quantize_16bit
        q_dw, scale = quantize_16bit(dw)
        M = np.ones(len(dw), dtype=bool)
        enc_vals = np.where(M, q_dw, 0.0)
        plain_vals = np.where(M, 0.0, q_dw)
        return Payload(cid, n_examples, plain_vals, he.encrypt(enc_vals), M, "batchcrypt", scale=scale)
    if getattr(cfg, "full_ckks", False):
        M = np.ones(len(dw), dtype=bool)
        enc_vals = np.where(M, dw, 0.0)
        plain_vals = np.where(M, 0.0, dw)
        return Payload(cid, n_examples, plain_vals, he.encrypt(enc_vals), M, "full_ckks")
    if not cfg.use_gashe:
        return Payload(cid, n_examples, plain=dw)
    M, _, tier = gashe_mask(dw, cfg)
    # np.where, NOT multiplication: 0 * NaN = NaN would poison BOTH streams
    enc_vals = np.where(M, dw, 0.0)
    plain_vals = np.where(M, 0.0, dw)
    return Payload(cid, n_examples, plain_vals, he.encrypt(enc_vals), M, tier)
