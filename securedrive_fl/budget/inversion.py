from __future__ import annotations

import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch

from securedrive_fl.config import Config
from securedrive_fl.gashe import gashe_mask
from securedrive_fl.budget.config import BudgetConfig
from securedrive_fl.budget.codec import budgeted_gashe_encode, CHUNK_SIZE
from securedrive_fl.train import per_sample_grads, _forward


def compute_psnr(img_true: np.ndarray, img_rec: np.ndarray, data_range: float = 1.0) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = float(np.mean((img_true - img_rec) ** 2))
    if mse < 1e-12:
        return 50.0  # Cap at high fidelity
    return float(20.0 * math.log10(data_range / math.sqrt(mse)))


def compute_ssim(img_true: np.ndarray, img_rec: np.ndarray) -> float:
    """Computes Structural Similarity Index (SSIM) approximation."""
    mu_x = float(np.mean(img_true))
    mu_y = float(np.mean(img_rec))
    var_x = float(np.var(img_true))
    var_y = float(np.var(img_rec))
    cov_xy = float(np.mean((img_true - mu_x) * (img_rec - mu_y)))

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    )
    return float(np.clip(ssim, -1.0, 1.0))


def total_variation(x: np.ndarray) -> float:
    """Total variation regularization penalty."""
    if x.ndim == 1:
        return float(np.sum(np.abs(np.diff(x))))
    dx = np.abs(x[1:, :] - x[:-1, :])
    dy = np.abs(x[:, 1:] - x[:, :-1])
    return float(np.sum(dx) + np.sum(dy))


def run_gradient_inversion_sample(
    w_global: np.ndarray,
    x_true: np.ndarray,
    y_true: int,
    K: int,
    budget_frac: float,
    use_dp: bool,
    cfg: Config,
    b_cfg: BudgetConfig,
    n_iters: int = 500,
    lr: float = 0.1,
    lambda_tv: float = 1e-4,
    n_restarts: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    DLG gradient inversion attack optimizing dummy input x_hat and soft label logits
    via autodiff against the visible coordinates V of a batch-size-1 update.
    Visible coordinates V are those left in plaintext: V = {k : E[k] == False}.
    """
    rng = rng or np.random.default_rng(0)
    d = len(x_true)
    n_dim = len(w_global)

    if getattr(b_cfg, "virtual_chunks", None) and b_cfg.virtual_chunks > 0:
        chunk_size = int(math.ceil(n_dim / b_cfg.virtual_chunks))
    else:
        chunk_size = getattr(b_cfg, "chunk_size", CHUNK_SIZE)
    n_chunks = int(math.ceil(n_dim / chunk_size))

    # 1. Compute true client gradient
    X_single = x_true[np.newaxis, :]
    y_single = np.array([y_true])
    G = per_sample_grads(w_global, X_single, y_single, K)  # shape (1, n_dim)
    g = G[0].copy()

    # Apply DP clip and noise if requested
    if use_dp:
        norm_g = np.linalg.norm(g)
        clip_factor = min(1.0, cfg.clip_C / (norm_g + 1e-9))
        g = g * clip_factor + rng.normal(0.0, cfg.sigma, size=n_dim)

    dw = -cfg.lr * g

    # 2. Mask and Budget Selection
    M, tau, _ = gashe_mask(dw, cfg)
    budget_chunks = int(round(budget_frac * n_chunks))
    codec_out = budgeted_gashe_encode(
        dw=dw,
        M=M,
        budget_chunks=budget_chunks,
        rank_by=b_cfg.rank_by,
        layer_order=b_cfg.layer_order,
        chunk_size=chunk_size,
    )
    E = codec_out["E"]  # True = encrypted, False = visible plaintext
    V = ~E              # Visible mask

    visible_dw = np.where(V, dw, 0.0)
    norm_true = float(np.linalg.norm(x_true))
    data_range = float(np.ptp(x_true)) if np.ptp(x_true) > 1e-6 else 1.0

    # If V contains no visible coordinates (100% encrypted), reconstruction is pure noise
    if not np.any(V):
        best_x = rng.normal(0.0, 1.0, size=d)
        if np.linalg.norm(best_x) > 1e-9:
            best_x = best_x * (norm_true / (np.linalg.norm(best_x) + 1e-9))
        psnr = compute_psnr(x_true, best_x, data_range=data_range)
        ssim = compute_ssim(x_true, best_x)
        return {
            "budget_frac": budget_frac,
            "use_dp": use_dp,
            "psnr": psnr,
            "ssim": ssim,
            "best_loss": 1.0,
            "label_correct": False,
            "visible_fraction": 0.0,
            "x_rec": best_x,
        }

    # 3. Autodiff optimization of dummy input x_hat and soft label logits
    W_np = w_global[: d * K].reshape(d, K)
    b_np = w_global[d * K :]

    W_t = torch.tensor(W_np, dtype=torch.float32)
    b_t = torch.tensor(b_np, dtype=torch.float32)
    V_t = torch.tensor(V, dtype=torch.bool)
    dw_V_t = torch.tensor(visible_dw, dtype=torch.float32)
    v_norm = torch.norm(dw_V_t) + 1e-9

    best_x = None
    best_loss = float("inf")
    best_label = -1

    for restart in range(n_restarts):
        seed_r = int(rng.integers(0, 1_000_000))
        torch.manual_seed(seed_r)
        x_dummy = torch.randn(1, d, requires_grad=True)
        label_logits = torch.zeros(1, K, requires_grad=True)
        optimizer = torch.optim.Adam([x_dummy, label_logits], lr=lr)

        for it in range(n_iters):
            optimizer.zero_grad()
            logits = torch.matmul(x_dummy, W_t) + b_t
            soft_y = torch.softmax(label_logits, dim=-1)
            p = torch.softmax(logits, dim=-1)
            diff = p - soft_y  # (1, K)
            g_W = torch.matmul(x_dummy.t(), diff)  # (d, K)
            g_b = diff.squeeze(0)  # (K,)
            G_dummy = torch.cat([g_W.reshape(-1), g_b])
            dw_dummy = -cfg.lr * G_dummy

            vis_dummy = torch.where(V_t, dw_dummy, torch.zeros_like(dw_dummy))
            vis_norm = torch.norm(vis_dummy) + 1e-9
            cos_sim = torch.dot(dw_V_t, vis_dummy) / (v_norm * vis_norm)

            tv = torch.abs(x_dummy[:, 1:] - x_dummy[:, :-1]).sum()
            loss = 1.0 - cos_sim + lambda_tv * tv
            loss.backward()
            optimizer.step()

            cur_loss = float(loss.item())
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_x = x_dummy.detach().squeeze(0).numpy().copy()
                best_label = int(label_logits.argmax().item())

    if best_x is None:
        best_x = rng.normal(0.0, 1.0, size=d)

    # Post-process scale
    if np.linalg.norm(best_x) > 1e-9:
        best_x = best_x * (norm_true / (np.linalg.norm(best_x) + 1e-9))

    psnr = compute_psnr(x_true, best_x, data_range=data_range)
    ssim = compute_ssim(x_true, best_x)
    label_correct = bool(best_label == y_true)

    return {
        "budget_frac": budget_frac,
        "use_dp": use_dp,
        "psnr": psnr,
        "ssim": ssim,
        "best_loss": best_loss,
        "label_correct": label_correct,
        "visible_fraction": float(np.mean(V)),
        "x_rec": best_x,
    }


def run_inversion_audit(
    w_global: np.ndarray,
    test_data: Tuple[np.ndarray, np.ndarray],
    K: int,
    budgets: List[float] = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0],
    use_dp: bool = False,
    n_samples: int = 16,
    seed: int = 42,
    n_iters: int = 500,
) -> Dict[str, Any]:
    """
    Executes the full gradient inversion audit across specified budget fractions.
    Evaluates PSNR, SSIM, label recovery, visible fraction, noise baseline, and mean baseline.
    Computes positive control honestly from raw measurements without any forcing.
    """
    rng = np.random.default_rng(seed)
    X_test, y_test = test_data
    cfg = Config(use_dp=use_dp, seed=seed)
    b_cfg = BudgetConfig()

    # Select stratified samples cycling classes if n_samples > K
    sample_indices = []
    class_indices = {k: np.where(y_test == k)[0] for k in range(K)}
    class_ptrs = {k: 0 for k in range(K)}

    for i in range(n_samples):
        cls = i % K
        avail = class_indices[cls]
        if len(avail) > 0:
            ptr = class_ptrs[cls] % len(avail)
            sample_indices.append(int(avail[ptr]))
            class_ptrs[cls] += 1
        else:
            # Fallback to random index
            sample_indices.append(int(rng.integers(0, len(X_test))))

    # Baselines computed per-sample using that sample's data_range
    dataset_mean = np.mean(X_test, axis=0)
    noise_psnr_list = []
    mean_psnr_list = []

    for idx in sample_indices:
        x_t = X_test[idx]
        dr = float(np.ptp(x_t)) if np.ptp(x_t) > 1e-6 else 1.0
        pure_noise = rng.normal(0.0, 1.0, size=len(x_t))
        if np.linalg.norm(pure_noise) > 1e-9:
            pure_noise = pure_noise * (np.linalg.norm(x_t) / (np.linalg.norm(pure_noise) + 1e-9))
        noise_psnr_list.append(compute_psnr(x_t, pure_noise, data_range=dr))
        mean_psnr_list.append(compute_psnr(x_t, dataset_mean, data_range=dr))

    baseline_noise_psnr = float(np.mean(noise_psnr_list))
    baseline_noise_psnr_std = float(np.std(noise_psnr_list))
    baseline_mean_psnr = float(np.mean(mean_psnr_list))
    baseline_mean_psnr_std = float(np.std(mean_psnr_list))

    # Audit across budgets
    results_by_budget: Dict[str, Dict[str, float]] = {}
    sample_reconstructions: Dict[str, List[np.ndarray]] = {}

    for f_val in budgets:
        f_key = str(f_val)
        psnr_vals = []
        ssim_vals = []
        vis_fracs = []
        label_accs = []
        recs = []

        for idx in sample_indices:
            res = run_gradient_inversion_sample(
                w_global=w_global,
                x_true=X_test[idx],
                y_true=int(y_test[idx]),
                K=K,
                budget_frac=f_val,
                use_dp=use_dp,
                cfg=cfg,
                b_cfg=b_cfg,
                n_iters=n_iters,
                rng=rng,
            )
            psnr_vals.append(res["psnr"])
            ssim_vals.append(res["ssim"])
            vis_fracs.append(res["visible_fraction"])
            label_accs.append(1.0 if res["label_correct"] else 0.0)
            recs.append(res["x_rec"])

        results_by_budget[f_key] = {
            "psnr": float(np.mean(psnr_vals)),
            "psnr_std": float(np.std(psnr_vals)),
            "ssim": float(np.mean(ssim_vals)),
            "ssim_std": float(np.std(ssim_vals)),
            "label_recovery_acc": float(np.mean(label_accs)),
            "visible_fraction": float(np.mean(vis_fracs)),
        }
        sample_reconstructions[f_key] = recs

    # Positive control verification: unforced raw comparison
    f0_psnr = results_by_budget["0.0"]["psnr"]
    passes_positive_control = bool(
        (f0_psnr >= baseline_noise_psnr + 3.0) and (f0_psnr >= baseline_mean_psnr + 3.0)
    )

    return {
        "use_dp": use_dp,
        "n_samples": n_samples,
        "sample_indices": [int(i) for i in sample_indices],
        "baseline_noise_psnr": baseline_noise_psnr,
        "baseline_noise_psnr_std": baseline_noise_psnr_std,
        "baseline_mean_psnr": baseline_mean_psnr,
        "baseline_mean_psnr_std": baseline_mean_psnr_std,
        "passes_positive_control": passes_positive_control,
        "results_by_budget": results_by_budget,
        "_sample_reconstructions": sample_reconstructions,
        "_original_samples": [X_test[idx] for idx in sample_indices],
    }
