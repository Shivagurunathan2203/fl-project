from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from securedrive_fl.he import HEBackend
from securedrive_fl.gashe import Payload


@dataclass
class Pending:
    """What the server queues for clients after a GASHE round."""
    c_agg: Any
    p_agg: np.ndarray
    n: int
    w_before: np.ndarray        # PRE-round weights (using post-round weights double-applies the update)
    scale: Any = None


def gashe_aggregate(payloads: list[Payload], he: HEBackend, w_before: np.ndarray) -> Pending:
    """Homomorphic addition only. No individual c_i is ever decrypted here."""
    c_agg = payloads[0].ciphertext
    for p in payloads[1:]:
        c_agg = he.add(c_agg, p.ciphertext)
    p_agg = np.sum([p.plain for p in payloads], axis=0)
    scale = getattr(payloads[0], "scale", None)
    return Pending(c_agg, p_agg, len(payloads), w_before.copy(), scale=scale)


def client_apply_update(he: HEBackend, pend: Pending) -> np.ndarray:
    """Runs on the client (holder of sk):  W <- W_before + (Dec(C_agg) + P_agg)/N."""
    dec_c = he.decrypt(pend.c_agg)
    if getattr(pend, "scale", None) is not None:
        dec_c = dec_c / pend.scale
    return pend.w_before + (dec_c + pend.p_agg) / pend.n


def fedavg(gw: np.ndarray, updates: list[np.ndarray], weights: list[int | float]) -> np.ndarray:
    w = np.asarray(weights, float)
    return gw + np.tensordot(w / w.sum(), np.stack(updates), axes=(0, 0))


def trimmed_mean(gw: np.ndarray, updates: list[np.ndarray], trim: float = 0.34) -> np.ndarray:
    # Yin et al. 2018; unweighted
    S = np.sort(np.stack(updates), axis=0)
    k = int(len(updates) * trim)
    return gw + S[k: len(updates) - k].mean(0) if len(updates) - 2 * k >= 1 else gw + S.mean(0)


def multi_krum(gw: np.ndarray, updates: list[np.ndarray], f: int = 1) -> np.ndarray:
    # Blanchard et al. 2017; unweighted
    n = len(updates)
    m = max(1, n - f - 2)
    scores = [sum(sorted(float(np.sum((u - v) ** 2)) for j, v in enumerate(updates) if j != i)[: n - f - 2])
              for i, u in enumerate(updates)]
    return gw + np.mean([updates[i] for i in np.argsort(scores)[:m]], axis=0)
