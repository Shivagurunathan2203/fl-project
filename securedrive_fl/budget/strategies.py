from __future__ import annotations

import math
import json
import time
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np

from securedrive_fl.config import Config
from securedrive_fl.he import HEBackend
from securedrive_fl.gashe import gashe_mask
from securedrive_fl.guard import LossGuard
from securedrive_fl.privacy import worst_case_epsilon
from securedrive_fl.train import local_train, evaluate, evaluate_full, n_params
from securedrive_fl.attacks import mitm_gashe
from securedrive_fl.budget.config import BudgetConfig
from securedrive_fl.budget.model import BudgetModel
from securedrive_fl.budget.codec import budgeted_gashe_encode, CHUNK_SIZE
from securedrive_fl.budget.aggregate import (
    BudgetedPayload,
    BudgetedPending,
    budgeted_gashe_aggregate,
    client_apply_budgeted_update,
)
from securedrive_fl.budget.time_model import LatencyModel


def run_budgeted_scenario(
    cfg: Config,
    b_cfg: BudgetConfig,
    strategy: str,
    client_data: List[Tuple[np.ndarray, np.ndarray]],
    test: Tuple[np.ndarray, np.ndarray],
    K: int,
    he: HEBackend,
    latency_model: Optional[LatencyModel] = None,
) -> Dict[str, Any]:
    """
    Executes a federated learning scenario under the compute-budgeted selective encryption model.
    Supports strategies:
      - 'sd_budget'
      - 'sd_wait'
      - 'sd_exclude'
      - 'sd_budget_random'
      - 'sd_budget_layer'
      - 'gashe_budget'
    """
    latency = latency_model or LatencyModel()
    rng = np.random.default_rng(cfg.seed)
    Xte, yte = test
    d = Xte.shape[1]
    n_dim = n_params(d, K)
    w = np.zeros(n_dim, dtype=np.float64)
    # Determine chunk size and virtual chunking
    if getattr(b_cfg, "virtual_chunks", None) and b_cfg.virtual_chunks > 0:
        chunk_size = int(math.ceil(n_dim / b_cfg.virtual_chunks))
        n_chunks = int(math.ceil(n_dim / chunk_size))
        virtual = True
    else:
        chunk_size = b_cfg.chunk_size
        n_chunks = int(math.ceil(n_dim / chunk_size))
        virtual = False
    n_clients = len(client_data)
    sizes = [len(X) for X, _ in client_data]

    # Instantiate separate BudgetModel with its own RNG stream
    b_model = BudgetModel(b_cfg, n_clients=n_clients, n_chunks=n_chunks, seed=cfg.seed)

    # Poisoning & guards
    poisoners = set(range(cfg.n_byzantine)) if cfg.attack == "model_poison" else set()
    guard = LossGuard() if cfg.loss_guard else None
    pending: Optional[BudgetedPending] = None

    # Determine rank_by for strategy
    rank_by = b_cfg.rank_by
    if strategy == "sd_budget_random":
        rank_by = "random"
    elif strategy == "sd_budget_layer":
        rank_by = "layer_static"

    # Tracking
    participation_counts = {cid: 0 for cid in range(n_clients)}
    rounds_log: List[Dict[str, Any]] = []
    total_sim_time = 0.0
    demand_fraction_ema = 1.0

    # Baseline training time per client per round (measured or fallback)
    t_train_nominal = 0.025 if cfg.use_dp else 0.010

    for r in range(1, cfg.rounds + 1):
        # 1. Apply pending update from previous round if present
        w_start = client_apply_budgeted_update(he, pending) if pending is not None else w
        pending = None
        w_before = w_start.copy()

        # 2. Get client effective budgets and speeds for round r
        eff_budgets = b_model.effective_chunks(r)
        speeds = b_model.speed(r)

        # 3. Determine active participants
        all_cids = list(range(n_clients))
        if strategy == "sd_exclude":
            # Client eligible only if B_{i,n} >= ceil(min_coverage * D_est)
            # D_est = ceil(demand_fraction_ema * n_chunks)
            D_est = int(math.ceil(demand_fraction_ema * n_chunks))
            req_chunks = int(math.ceil(getattr(b_cfg, "min_coverage", 0.5) * D_est))
            eligible = [cid for cid in all_cids if eff_budgets[cid] >= req_chunks]

            if cfg.attack == "model_poison":
                # Maintain up to cfg.n_byzantine poisoners among eligible
                eligible_poisoners = [cid for cid in eligible if cid in poisoners]
                eligible_benign = [cid for cid in eligible if cid not in poisoners]
                max_benign = max(0, len(client_data) - len(eligible_poisoners))
                active_cids = eligible_poisoners[:cfg.n_byzantine] + eligible_benign[:max_benign]
            else:
                active_cids = eligible[:len(client_data)]

            if len(active_cids) == 0:
                # Skip round if no eligible clients
                acc, loss = evaluate(w_start, Xte, yte, K)
                m = evaluate_full(w_start, Xte, yte, K, target_class=getattr(cfg, "target_class", 2))
                rounds_log.append({
                    "round": r,
                    "n_participants": 0,
                    "mean_budget_chunks": 0.0,
                    "mean_speed": 0.0,
                    "enc_chunks_total": 0,
                    "enc_chunk_fraction": 0.0,
                    "demand_fraction": 0.0,
                    "protected_energy_fraction": 0.0,
                    "protected_coord_fraction": 0.0,
                    "plain_bytes_per_client": 0,
                    "cipher_bytes_per_client": 0,
                    "sim_round_time_s": 0.0,
                    "skipped_round": True,
                    "speeds": speeds.tolist(),
                    "budgets": eff_budgets.tolist(),
                    "selected_chunk_counts": [0] * n_clients,
                    "client_protected_energies": {},
                    "participated": [False] * n_clients,
                    "acc": m["acc"],
                    "loss": m["loss"],
                    "asr": m["asr"],
                    "macro_f1": m["macro_f1"],
                    "macro_precision": m["macro_precision"],
                    "rejected": False,
                    "flagged": False,
                    "weights": w_start.copy(),
                    "eps": worst_case_epsilon(sizes, r, cfg) if cfg.use_dp else None,
                })
                w = w_start
                continue
        else:
            active_cids = all_cids

        # Mark participants
        participated_flags = [False] * n_clients
        for cid in active_cids:
            participated_flags[cid] = True
            participation_counts[cid] += 1

        # 4. Local training + Budgeted encoding
        payloads: List[BudgetedPayload] = []
        demand_counts: List[int] = []
        enc_counts: List[int] = []
        selected_chunk_counts = [0] * n_clients

        for cid in active_cids:
            X, y = client_data[cid]
            is_poison = (cid in poisoners)
            is_ascent = is_poison and (getattr(cfg, "poison_mode", "ascent") == "ascent")
            dw = local_train(w_start, X, y, K, cfg, rng, ascent=is_ascent, poison=is_poison)

            # GASHE threshold mask M_i
            M, tau, tier = gashe_mask(dw, cfg)

            # Client budget
            b_cid = eff_budgets[cid]
            if strategy in ("sd_full", "sd_wait"):
                # Unlimited budget
                b_cid = n_chunks

            # Encode
            rng_client = np.random.default_rng(cfg.seed + b_cfg.rng_salt + cid * 1009 + r * 503)
            enc_res = budgeted_gashe_encode(
                dw=dw,
                M=M,
                budget_chunks=b_cid,
                rank_by=rank_by,
                layer_order=b_cfg.layer_order,
                chunk_content=b_cfg.chunk_content,
                chunk_size=chunk_size,
                rng=rng_client,
            )

            demand_counts.append(len(enc_res["demand_set"]))
            enc_counts.append(len(enc_res["selected_set"]))
            selected_chunk_counts[cid] = len(enc_res["selected_set"])

            # HE encryption for selected chunk vectors
            cipher_chunks: Dict[int, Any] = {}
            for j, v_j in enc_res["enc_chunk_vectors"].items():
                cipher_chunks[j] = he.encrypt(v_j)

            payload = BudgetedPayload(
                cid=cid,
                plain=enc_res["plain_stream"],
                cipher_chunks=cipher_chunks,
                selection_bitmap=enc_res["selection_bitmap"],
                M=enc_res["M"],
                E=enc_res["E"],
                stats=enc_res,
                n_examples=len(X),
                chunk_size=chunk_size,
            )
            payloads.append(payload)

        # Update demand EMA for sd_exclude
        if payloads:
            observed_demand_frac = float(np.mean([p.stats["demand_fraction"] for p in payloads]))
            demand_fraction_ema = 0.5 * demand_fraction_ema + 0.5 * observed_demand_frac

        # Arrival order is nondeterministic -> sort by REAL cid before any indexing
        rng.shuffle(payloads)
        payloads.sort(key=lambda p: p.cid)

        # 5. Attack hook: MitM on intercepted payloads (shared with base)
        if cfg.attack == "mitm":
            mitm_gashe(payloads, he, rng, cfg)

        # 6. Aggregation
        pending = budgeted_gashe_aggregate(payloads, he, w_before, chunk_size=chunk_size)
        w = client_apply_budgeted_update(he, pending)

        # 7. LossGuard detection and rollback
        acc, loss = evaluate(w, Xte, yte, K)
        rejected = bool(guard and guard.reject(loss))
        if rejected:
            w, pending = w_before, None
            acc, loss = evaluate(w, Xte, yte, K)

        m = evaluate_full(w, Xte, yte, K, target_class=getattr(cfg, "target_class", 2))

        # 8. Timing & Resource Logging
        unique_enc_chunks = len(set.union(*[p.stats["selected_set"] for p in payloads])) if payloads else 0
        part_speeds = [speeds[cid] for cid in active_cids]
        sim_round_time = latency.compute_round_time(
            strategy=strategy,
            t_train=t_train_nominal,
            n_chunks=n_chunks,
            client_speeds=part_speeds,
            demand_counts=demand_counts,
            encrypted_counts=enc_counts,
            total_unique_enc_chunks=unique_enc_chunks,
        )
        total_sim_time += sim_round_time

        # Upload byte metrics per client
        mean_plain_bytes = int(np.mean([np.count_nonzero(p.plain) * latency.plain_bytes_per_coord for p in payloads]))
        mean_cipher_bytes = int(np.mean([len(p.cipher_chunks) * latency.bytes_per_chunk for p in payloads]))

        rounds_log.append({
            "round": r,
            "n_participants": len(active_cids),
            "mean_budget_chunks": float(np.mean([eff_budgets[cid] for cid in active_cids])),
            "mean_speed": float(np.mean(part_speeds)),
            "enc_chunks_total": int(sum(enc_counts)),
            "enc_chunk_fraction": float(np.mean([p.stats["enc_chunk_fraction"] for p in payloads])),
            "demand_fraction": float(np.mean([p.stats["demand_fraction"] for p in payloads])),
            "protected_energy_fraction": float(np.mean([p.stats["protected_energy_fraction"] for p in payloads])),
            "protected_coord_fraction": float(np.mean([p.stats["protected_coord_fraction"] for p in payloads])),
            "plain_bytes_per_client": mean_plain_bytes,
            "cipher_bytes_per_client": mean_cipher_bytes,
            "sim_round_time_s": sim_round_time,
            "skipped_round": False,
            "speeds": speeds.tolist(),
            "budgets": eff_budgets.tolist(),
            "selected_chunk_counts": selected_chunk_counts,
            "client_protected_energies": {p.cid: float(p.stats["protected_energy_fraction"]) for p in payloads},
            "participated": participated_flags,
            "acc": m["acc"],
            "loss": m["loss"],
            "asr": m["asr"],
            "macro_f1": m["macro_f1"],
            "macro_precision": m["macro_precision"],
            "rejected": rejected,
            "flagged": rejected,
            "weights": w.copy(),
            "eps": worst_case_epsilon(sizes, r, cfg) if cfg.use_dp else None,
        })

    # Summary metrics
    valid_rounds = [x for x in rounds_log if not x["skipped_round"]]
    if valid_rounds:
        mean_prot_energy = float(np.mean([x["protected_energy_fraction"] for x in valid_rounds]))
        mean_prot_coords = float(np.mean([x["protected_coord_fraction"] for x in valid_rounds]))
        mean_eff_budget = float(np.mean([x["mean_budget_chunks"] for x in valid_rounds]))
    else:
        mean_prot_energy = 0.0
        mean_prot_coords = 0.0
        mean_eff_budget = 0.0

    min_eff_budget = int(min(b_model.effective_chunks(r).min() for r in range(1, cfg.rounds + 1)))
    max_eff_budget = int(max(b_model.effective_chunks(r).max() for r in range(1, cfg.rounds + 1)))

    total_parts = sum(participation_counts.values())
    degenerate = bool(total_parts == 0)

    final_eps = worst_case_epsilon(sizes, cfg.rounds, cfg) if cfg.use_dp else None
    eps_120 = worst_case_epsilon(sizes, 120, cfg) if cfg.use_dp else None

    return {
        "strategy": strategy,
        "scenario": b_cfg.capacity.mode,
        "rank_by": rank_by,
        "chunk_content": b_cfg.chunk_content,
        "chunk_size": chunk_size,
        "n_chunks": n_chunks,
        "virtual": virtual,
        "min_coverage": getattr(b_cfg, "min_coverage", 0.5),
        "degenerate": degenerate,
        "capacity": b_model.capacity.tolist(),
        "mean_effective_budget": mean_eff_budget,
        "min_effective_budget": min_eff_budget,
        "max_effective_budget": max_eff_budget,
        "total_sim_time_s": total_sim_time,
        "mean_protected_energy_fraction": mean_prot_energy,
        "mean_protected_coord_fraction": mean_prot_coords,
        "participation_counts": participation_counts,
        "final_accuracy": rounds_log[-1]["acc"],
        "macro_f1": rounds_log[-1]["macro_f1"],
        "macro_precision": rounds_log[-1]["macro_precision"],
        "asr": rounds_log[-1]["asr"],
        "final_epsilon": final_eps,
        "epsilon_at_120": eps_120,
        "rounds": rounds_log,
        "final_weights": w,
    }
