from __future__ import annotations

import math
from typing import Dict, Any, List, Set, Tuple, Optional
import numpy as np


CHUNK_SIZE = 4096


def get_chunk_slice(j: int, n_chunks: int, d: int, chunk_size: int = CHUNK_SIZE) -> slice:
    start = j * chunk_size
    end = min(start + chunk_size, d)
    return slice(start, end)


def budgeted_gashe_encode(
    dw: np.ndarray,
    M: np.ndarray,
    budget_chunks: int,
    rank_by: str = "post_noise_energy",
    layer_order: str = "reverse",
    chunk_content: str = "masked_only",
    chunk_size: int = CHUNK_SIZE,
    rng: Optional[np.random.Generator] = None,
    pre_noise_dw: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Client-side budgeted selective encryption codec.

    Partitions coordinate space into 4096-sized chunks, identifies demand chunks,
    ranks demand chunks according to `rank_by`, and selects top B chunks for HE.
    The remainder of coordinates (including unselected high-magnitude ones) are
    routed to the plaintext stream.
    """
    d = len(dw)
    n_chunks = int(math.ceil(d / chunk_size))

    # 1. Demand set D_i = { j : M has at least one 1 in chunk j }
    demand_chunks: List[int] = []
    for j in range(n_chunks):
        s = get_chunk_slice(j, n_chunks, d, chunk_size)
        if M[s].any():
            demand_chunks.append(j)

    n_demand = len(demand_chunks)
    n_select = min(max(0, int(budget_chunks)), n_demand)

    # 2. Score demand chunks
    scored_chunks: List[Tuple[float, int]] = []
    if rank_by == "post_noise_energy":
        for j in demand_chunks:
            s = get_chunk_slice(j, n_chunks, d, chunk_size)
            chunk_m = M[s]
            chunk_w = dw[s]
            energy = float(np.sum((chunk_m * chunk_w) ** 2))
            # Tie breaking by lower chunk index: tuple (-energy, j)
            scored_chunks.append((-energy, j))
        scored_chunks.sort()
        selected_set: Set[int] = {j for _, j in scored_chunks[:n_select]}

    elif rank_by == "layer_static":
        for j in demand_chunks:
            # reverse: later chunks (e.g. classifier) get highest priority
            score = float(j if layer_order == "reverse" else (n_chunks - j))
            scored_chunks.append((-score, j))
        scored_chunks.sort()
        selected_set = {j for _, j in scored_chunks[:n_select]}

    elif rank_by == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        perm = rng.permutation(demand_chunks).tolist()
        selected_set = set(perm[:n_select])

    elif rank_by == "pre_noise_energy":
        src = pre_noise_dw if pre_noise_dw is not None else dw
        for j in demand_chunks:
            s = get_chunk_slice(j, n_chunks, d, chunk_size)
            chunk_m = M[s]
            chunk_w = src[s]
            energy = float(np.sum((chunk_m * chunk_w) ** 2))
            scored_chunks.append((-energy, j))
        scored_chunks.sort()
        selected_set = {j for _, j in scored_chunks[:n_select]}

    else:
        raise ValueError(f"Unknown rank_by mode: {rank_by}")

    # 3. Construct encrypted coordinate mask E_i
    E = np.zeros(d, dtype=bool)
    selection_bitmap = np.zeros(n_chunks, dtype=bool)
    for j in selected_set:
        selection_bitmap[j] = True
        s = get_chunk_slice(j, n_chunks, d, chunk_size)
        if chunk_content == "full_chunk":
            E[s] = True
        else:
            E[s] = M[s]

    # 4. Prepare chunk payload vectors for HE
    # Maps chunk_idx -> vector of coordinates for that chunk
    enc_chunk_vectors: Dict[int, np.ndarray] = {}
    for j in selected_set:
        s = get_chunk_slice(j, n_chunks, d, chunk_size)
        if chunk_content == "masked_only":
            # Masked-only: E_i \odot dw inside chunk j
            enc_chunk_vectors[j] = np.where(E[s], dw[s], 0.0)
        elif chunk_content == "full_chunk":
            enc_chunk_vectors[j] = dw[s].copy()
        else:
            raise ValueError(f"Unknown chunk_content mode: {chunk_content}")

    # 5. Plaintext stream: (1 - E_i) \odot dw
    plain_stream = np.where(~E, dw, 0.0)

    # 6. Statistics
    total_energy = float(np.sum(dw ** 2)) + 1e-12
    protected_energy = float(np.sum((E * dw) ** 2))
    protected_energy_fraction = protected_energy / total_energy
    protected_coord_fraction = float(np.mean(E))
    demand_fraction = float(n_demand / max(1, n_chunks))
    enc_chunk_fraction = float(len(selected_set) / max(1, n_chunks))

    return {
        "enc_chunk_vectors": enc_chunk_vectors,
        "plain_stream": plain_stream,
        "M": M,
        "E": E,
        "selection_bitmap": selection_bitmap,
        "demand_set": set(demand_chunks),
        "selected_set": selected_set,
        "n_chunks": n_chunks,
        "demand_fraction": demand_fraction,
        "enc_chunk_fraction": enc_chunk_fraction,
        "protected_energy_fraction": protected_energy_fraction,
        "protected_coord_fraction": protected_coord_fraction,
    }
