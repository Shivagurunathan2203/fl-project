from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np

from securedrive_fl.he import HEBackend
from securedrive_fl.budget.codec import CHUNK_SIZE, get_chunk_slice


@dataclass
class BudgetedPayload:
    cid: int
    plain: np.ndarray
    cipher_chunks: Dict[int, Any]  # chunk_idx -> ciphertext
    selection_bitmap: np.ndarray
    M: np.ndarray
    E: np.ndarray
    stats: Dict[str, Any]
    n_examples: int = 1
    chunk_size: int = CHUNK_SIZE


@dataclass
class BudgetedPending:
    c_agg_chunks: Dict[int, Any]  # chunk_idx -> aggregated ciphertext
    p_agg: np.ndarray             # aggregated plaintext stream
    n: int                        # number of participating clients
    w_before: np.ndarray          # model weights before this round
    d: int                        # total dimension
    n_chunks: int                 # total chunk count
    chunk_size: int = CHUNK_SIZE


def budgeted_gashe_aggregate(
    payloads: List[BudgetedPayload],
    he: HEBackend,
    w_before: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
) -> BudgetedPending:
    """
    Server-side homomorphic aggregation over budgeted chunk payloads.
    Server possesses only public keys (cannot decrypt).
    """
    if not payloads:
        d = len(w_before)
        n_chunks = int(math.ceil(d / chunk_size))
        return BudgetedPending(
            c_agg_chunks={},
            p_agg=np.zeros(d, dtype=np.float64),
            n=0,
            w_before=w_before,
            d=d,
            n_chunks=n_chunks,
            chunk_size=chunk_size,
        )

    d = len(payloads[0].plain)
    n_chunks = int(math.ceil(d / chunk_size))

    # 1. Sum plaintext streams
    p_agg = np.sum([p.plain for p in payloads], axis=0)

    # 2. Homomorphically sum encrypted chunks
    c_agg_chunks: Dict[int, Any] = {}
    for p in payloads:
        for j, c_j in p.cipher_chunks.items():
            if j not in c_agg_chunks:
                c_agg_chunks[j] = c_j
            else:
                c_agg_chunks[j] = he.add(c_agg_chunks[j], c_j)

    return BudgetedPending(
        c_agg_chunks=c_agg_chunks,
        p_agg=p_agg,
        n=len(payloads),
        w_before=w_before,
        d=d,
        n_chunks=n_chunks,
        chunk_size=chunk_size,
    )


def client_apply_budgeted_update(
    he: HEBackend,
    pending: BudgetedPending,
) -> np.ndarray:
    """
    Client-side decryption and weight update.
    Reconstructs: ΔW_full = scatter(Dec(C_agg)) + P_agg
    W_new = w_before + ΔW_full / n
    """
    d = pending.d
    if pending.n == 0:
        return pending.w_before.copy()

    # 1. Decrypt each chunk in C_agg and scatter into flat array
    dw_enc = np.zeros(d, dtype=np.float64)
    for j, c_j in pending.c_agg_chunks.items():
        s = get_chunk_slice(j, pending.n_chunks, d, pending.chunk_size)
        dec_vals = he.decrypt(c_j)
        chunk_len = s.stop - s.start
        # Handle TenSEAL or Mock format
        if isinstance(dec_vals, (list, tuple)):
            dec_vals = np.array(dec_vals, dtype=np.float64)
        dw_enc[s] = dec_vals[:chunk_len]

    # 2. Reconstruct full update
    dw_full = dw_enc + pending.p_agg

    # 3. Apply normalized step
    return pending.w_before + (dw_full / float(pending.n))
