from __future__ import annotations

import math
from typing import Any, List
import numpy as np
from securedrive_fl.config import Config
from securedrive_fl.he import HEBackend


def get_chunk_slice(j: int, n_chunks: int, d: int, chunk_size: int = 4096) -> slice:
    start = j * chunk_size
    end = min(start + chunk_size, d)
    return slice(start, end)


def mitm_plain(updates: list[np.ndarray], cfg: Config, rng) -> list[np.ndarray]:
    """No-GASHE channel: first f=N/3 clients get  -amplify*u + N(0, std²)  (Blanchard 2017)."""
    out = [u.copy() for u in updates]
    noise_std = cfg.mitm_noise_std
    for i in range(cfg.n_byzantine):
        out[i] = -cfg.mitm_amplify * updates[i] + rng.normal(0, noise_std, updates[i].shape)
    return out


def mitm_gashe(payloads: List[Any], he: HEBackend, rng, cfg: Config) -> None:
    """
    GASHE channel (in place). The attacker cannot read or sensibly edit a
    ciphertext, but CKKS gives confidentiality, not integrity, so it can
    SUBSTITUTE one: discard c_i and encrypt fresh noise under the public key.
    The plaintext stream is perturbed with noise at the same observed scale.
    Supports both base Payload and BudgetedPayload.
    """
    for p in payloads[: cfg.n_byzantine]:                # same persistent victims every round
        if hasattr(p, "E") and p.E is not None:
            seen = p.plain[~p.E]
        elif getattr(p, "mask", None) is not None:
            seen = p.plain[p.mask == 0]                  # observable plaintext coordinates
        else:
            seen = p.plain
        std = float(np.std(seen[seen != 0])) + 1e-8 if (seen is not None and len(seen) > 0 and (seen != 0).any()) else 1e-8

        if hasattr(p, "cipher_chunks") and p.cipher_chunks is not None:
            d = len(p.plain)
            chunk_size = getattr(p, "chunk_size", getattr(cfg, "chunk_size", 4096))
            n_chunks = int(math.ceil(d / chunk_size))
            for j in sorted(p.cipher_chunks.keys()):
                s = get_chunk_slice(j, n_chunks, d, chunk_size)
                chunk_len = s.stop - s.start
                noise_chunk = rng.normal(0, std, chunk_len)
                p.cipher_chunks[j] = he.encrypt(noise_chunk)
        else:
            p.ciphertext = he.encrypt(rng.normal(0, std, len(p.plain)))

        p.plain = p.plain + rng.normal(0, std, p.plain.shape)


def stamp_backdoor_trigger(
    data: np.ndarray,
    patch_size: int = 4,
    trigger_val: float = 0.6,
) -> np.ndarray:
    """
    Stamps an 8x8 backdoor trigger on images, or a feature-space trigger
    (+trigger_val on the first patch_size coordinates) on feature vectors.
    Note: NumPy training uses feature-space triggers, not 8x8 image patches.
    """
    out = data.copy()
    if out.ndim == 4:
        # (B, H, W, C)
        h, w = out.shape[1], out.shape[2]
        ps = max(8, patch_size)
        out[:, h - ps:h, w - ps:w, :] = 1.0 if out.dtype in (np.float32, np.float64) else 255
    elif out.ndim == 3:
        # (H, W, C)
        h, w = out.shape[0], out.shape[1]
        ps = max(8, patch_size)
        out[h - ps:h, w - ps:w, :] = 1.0 if out.dtype in (np.float32, np.float64) else 255
    elif out.ndim == 2:
        # Batch of feature vectors (B, d)
        k = min(patch_size, out.shape[1])
        out[:, :k] += trigger_val
    elif out.ndim == 1:
        # Single feature vector (d,)
        k = min(patch_size, len(out))
        out[:k] += trigger_val
    return out
