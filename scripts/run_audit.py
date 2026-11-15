#!/usr/bin/env python3
"""
Reconstruction Audit (Gradient Inversion Leakage Evaluation).
Evaluates information leakage in the visible plaintext stream across budget fractions f.
Verifies positive control (spec_budget.md Section 7 / final_spec.txt Section 7).
"""

import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from securedrive_fl.config import Config
from securedrive_fl.he import MockHE
from securedrive_fl.data import make_toy_federation
from securedrive_fl.run_scenario import run_scenario
from securedrive_fl.budget.inversion import run_inversion_audit


def main():
    parser = argparse.ArgumentParser(description="Reconstruction Audit (DLG).")
    parser.add_argument("--rounds", type=int, default=30, help="Training rounds for base models (default: 30)")
    parser.add_argument("--n_samples", type=int, default=16, help="Number of test samples to audit (default: 16)")
    parser.add_argument("--n_iters", type=int, default=500, help="DLG optimization iterations per restart (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results-ext-dir", default="results_ext", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.results_ext_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "audit_results.json"
    out_npz = out_dir / "audit_samples.npz"

    if out_json.exists() and out_npz.exists():
        print(f"[Info] Audit results already exist at {out_json}. Skipping audit.")
        return

    print("==========================================================")
    print("Starting Reconstruction Audit (Gradient Inversion / DLG)")
    print(f"Rounds: {args.rounds} | Samples: {args.n_samples} | Iters: {args.n_iters} | Seed: {args.seed}")
    print("==========================================================")

    # Prepare data and trained base model
    client_data, test_data, K = make_toy_federation(seed=args.seed)
    he = MockHE()

    print(f"[Info] Training non-DP base model (FedAvg) for {args.rounds} rounds...")
    cfg_plain = Config(rounds=args.rounds, use_dp=False, use_gashe=False, seed=args.seed)
    hist_plain = run_scenario(cfg_plain, client_data, test_data, K, he)
    w_non_dp = hist_plain.final_weights
    print(f"  FedAvg base model final accuracy: {hist_plain[-1]['acc']*100:.2f}%")

    print(f"[Info] Training DP base model (DP-SGD) for {args.rounds} rounds...")
    cfg_dp = Config(rounds=args.rounds, use_dp=True, use_gashe=False, seed=args.seed)
    hist_dp = run_scenario(cfg_dp, client_data, test_data, K, he)
    w_dp = hist_dp.final_weights
    print(f"  DP-SGD base model final accuracy: {hist_dp[-1]['acc']*100:.2f}%")

    budgets = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]

    # 1. Non-DP Audit
    print(f"\n[Audit 1/2] Running Non-DP Reconstruction Audit across budgets {budgets}...")
    audit_non_dp = run_inversion_audit(
        w_global=w_non_dp,
        test_data=test_data,
        K=K,
        budgets=budgets,
        use_dp=False,
        n_samples=args.n_samples,
        seed=args.seed,
        n_iters=args.n_iters,
    )
    print(f"  Noise Baseline PSNR: {audit_non_dp['baseline_noise_psnr']:.2f} ± {audit_non_dp['baseline_noise_psnr_std']:.2f} dB")
    print(f"  Mean Baseline PSNR:  {audit_non_dp['baseline_mean_psnr']:.2f} ± {audit_non_dp['baseline_mean_psnr_std']:.2f} dB")
    print(f"  Positive Control:    {'PASSED' if audit_non_dp['passes_positive_control'] else 'FAILED'}")
    for f_val in budgets:
        f_key = str(f_val)
        stats = audit_non_dp["results_by_budget"][f_key]
        print(f"    f={f_key:<4} -> PSNR: {stats['psnr']:.2f} ± {stats['psnr_std']:.2f} dB, SSIM: {stats['ssim']:.3f}, LabelAcc: {stats['label_recovery_acc']*100:.1f}%, Visible: {stats['visible_fraction']*100:.1f}%")

    if not audit_non_dp["passes_positive_control"]:
        attempts_file = out_dir / "audit_attempts.md"
        with open(attempts_file, "w") as f:
            f.write(f"# Reconstruction Audit Attempts\n\nPositive control check failed with f=0 PSNR: {audit_non_dp['results_by_budget']['0.0']['psnr']:.2f} dB vs Noise: {audit_non_dp['baseline_noise_psnr']:.2f} dB and Mean: {audit_non_dp['baseline_mean_psnr']:.2f} dB.\n")
        print(f"[Warning] Positive control check failed; logged to {attempts_file}")

    # 2. DP Audit
    print(f"\n[Audit 2/2] Running DP (eps0=4) Reconstruction Audit across budgets...")
    audit_dp = run_inversion_audit(
        w_global=w_dp,
        test_data=test_data,
        K=K,
        budgets=budgets,
        use_dp=True,
        n_samples=args.n_samples,
        seed=args.seed,
        n_iters=args.n_iters,
    )
    for f_val in budgets:
        f_key = str(f_val)
        stats = audit_dp["results_by_budget"][f_key]
        print(f"    f={f_key:<4} -> PSNR: {stats['psnr']:.2f} ± {stats['psnr_std']:.2f} dB, SSIM: {stats['ssim']:.3f}, Visible: {stats['visible_fraction']*100:.1f}%")

    # Extract sample reconstructions for plotting Fig 4 example grids
    npz_dict = {}
    if "_sample_reconstructions" in audit_non_dp:
        for f_key, rec_list in audit_non_dp["_sample_reconstructions"].items():
            npz_dict[f"non_dp_rec_f{f_key}"] = np.stack(rec_list)
        del audit_non_dp["_sample_reconstructions"]
    if "_original_samples" in audit_non_dp:
        npz_dict["original_samples"] = np.stack(audit_non_dp["_original_samples"])
        del audit_non_dp["_original_samples"]

    if "_sample_reconstructions" in audit_dp:
        for f_key, rec_list in audit_dp["_sample_reconstructions"].items():
            npz_dict[f"dp_rec_f{f_key}"] = np.stack(rec_list)
        del audit_dp["_sample_reconstructions"]
    if "_original_samples" in audit_dp:
        del audit_dp["_original_samples"]

    np.savez_compressed(out_npz, **npz_dict)

    final_payload = {
        "budgets": budgets,
        "n_samples": args.n_samples,
        "non_dp_audit": audit_non_dp,
        "dp_audit": audit_dp,
    }

    with open(out_json, "w") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\n[Success] Audit results saved to {out_json} and samples to {out_npz}")


if __name__ == "__main__":
    main()
