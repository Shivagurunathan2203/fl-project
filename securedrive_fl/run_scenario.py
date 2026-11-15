from __future__ import annotations

import math
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from securedrive_fl.config import Config
from securedrive_fl.he import HEBackend
from securedrive_fl.gashe import Payload, make_payload
from securedrive_fl.aggregate import (
    Pending,
    gashe_aggregate,
    client_apply_update,
    fedavg,
    trimmed_mean,
    multi_krum,
)
from securedrive_fl.attacks import mitm_plain, mitm_gashe
from securedrive_fl.guard import LossGuard
from securedrive_fl.privacy import worst_case_epsilon
from securedrive_fl.train import local_train, evaluate, evaluate_full, n_params


class ScenarioHistory(list):
    """
    Subclass of list holding round history dictionaries, with attribute and key access
    to final_weights for backwards compatibility across callers.
    """
    def __init__(self, history_items: list, final_weights: np.ndarray):
        super().__init__(history_items)
        self.final_weights = final_weights
        self.w = final_weights

    def __getitem__(self, key):
        if key == "history":
            return list(self)
        if key == "final_weights":
            return self.final_weights
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "history":
            return list(self)
        if key == "final_weights":
            return self.final_weights
        return default


def run_scenario(
    cfg: Config,
    client_data: List[Tuple[np.ndarray, np.ndarray]],
    test: Tuple[np.ndarray, np.ndarray],
    K: int,
    he: HEBackend,
) -> ScenarioHistory:
    rng = np.random.default_rng(cfg.seed)
    Xte, yte = test
    d = Xte.shape[1]
    w = np.zeros(n_params(d, K))                         # server's global model
    sizes = [len(X) for X, _ in client_data]
    poisoners = set(range(cfg.n_byzantine)) if cfg.attack == "model_poison" else set()
    guard = LossGuard() if cfg.loss_guard else None
    pending: Optional[Pending] = None
    history = []

    for r in range(1, cfg.rounds + 1):
        # (a,b) what clients start from: pending GASHE update (decrypted by them) or plain weights
        w_start = client_apply_update(he, pending) if pending is not None else w
        pending = None

        # (c,d) local training + payload creation
        payloads = []
        for cid, (X, y) in enumerate(client_data):
            is_poison = (cid in poisoners)
            is_ascent = is_poison and (getattr(cfg, "poison_mode", "ascent") == "ascent")
            dw = local_train(w_start, X, y, K, cfg, rng, ascent=is_ascent, poison=is_poison)
            payloads.append(make_payload(cid, dw, len(X), cfg, he))

        # (e,f) arrival order is nondeterministic -> sort by REAL cid before any indexing
        rng.shuffle(payloads)
        payloads.sort(key=lambda p: p.cid)
        w_before = w.copy()

        # (g,h) attack hook + aggregation
        is_he_path = cfg.use_gashe or getattr(cfg, "full_ckks", False) or getattr(cfg, "batchcrypt", False)
        if is_he_path:
            if cfg.attack == "mitm":
                mitm_gashe(payloads, he, rng, cfg)
            pending = gashe_aggregate(payloads, he, w_before)
            # evaluation-only mirror of the clients' update (V56 does the same)
            w = client_apply_update(he, pending)
        else:
            updates = [p.plain for p in payloads]
            if cfg.attack == "mitm":
                updates = mitm_plain(updates, cfg, rng)
            weights = [p.n_examples for p in payloads]
            aggregator_map = {
                "fedavg": lambda: fedavg(w_before, updates, weights),
                "fedprox": lambda: fedavg(w_before, updates, weights),
                "trimmed_mean": lambda: trimmed_mean(w_before, updates),
                "krum": lambda: multi_krum(w_before, updates, f=getattr(cfg, "krum_f", 2)),
            }
            w = aggregator_map[cfg.aggregator]()

        # (i) loss guard: roll back AND drop the queued update, or clients would still apply it
        acc, loss = evaluate(w, Xte, yte, K)
        rejected = bool(guard and guard.reject(loss))
        if rejected:
            w, pending = w_before, None
            acc, loss = evaluate(w, Xte, yte, K)

        m = evaluate_full(w, Xte, yte, K, target_class=getattr(cfg, "target_class", 2))

        # (j) accounting + metrics
        enc_frac = (
            float(np.mean([p.mask.mean() for p in payloads]))
            if (is_he_path and payloads[0].mask is not None)
            else 0.0
        )
        history.append({
            "round": r,
            "acc": m["acc"],
            "loss": m["loss"],
            "asr": m["asr"],
            "macro_f1": m["macro_f1"],
            "macro_precision": m["macro_precision"],
            "rejected": rejected,
            "flagged": rejected,
            "weights": w.copy(),
            "enc_frac": enc_frac,                         # fraction of coords under HE
            "leakage": 1.0 - enc_frac if is_he_path else 1.0,  # plaintext share seen by an eavesdropper
            "eps": worst_case_epsilon(sizes, r, cfg) if cfg.use_dp else None,
            "tiers": [p.tier for p in payloads],
        })
    return ScenarioHistory(history, w)
