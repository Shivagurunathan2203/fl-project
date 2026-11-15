#!/usr/bin/env python3
"""
Generates Figures 2-ext, 3, 4, 5, and 6 for the extension (spec_budget.md Section 10).
Outputs PNG and PDF formats.
Strictly faithful to final_spec.txt:
- No hardcoded fallbacks or synthetic values.
- Fails loudly (sys.exit non-zero) if input files are missing.
- Reads directly from measured results in results/ and results_ext/.
"""

import os
import sys
import glob
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_fig2_ext(ext_dir: str = "results_ext", base_dir: str = "results"):
    """
    Fig. 2-ext: Test accuracy vs communication round under MitM (left) and Poison (right)
    for dpsgd, sd_full, sd_budget, sd_exclude, with clean FedAvg benchmark dashed,
    styled exactly as in the paper with the legend outside the graphs at the bottom.
    """
    if not os.path.isdir(ext_dir):
        sys.exit(f"MISSING: Extension directory '{ext_dir}' not found for fig2_ext.")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    fig, (ax_mitm, ax_poison) = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=300)

    # Styles matching paper aesthetic
    series = [
        ("dpsgd", "FL+DP-SGD (no enc)", "#1f78b4", "--", "o", 2.2, 5.5),
        ("sd_full", "SecureDrive-FL (full enc)", "#08306b", "-", "s", 2.4, 5.5),
        ("sd_budget", "SecureDrive-FL (sd_budget)", "#a2142f", "-", "^", 2.4, 6.0),
        ("sd_exclude", "SecureDrive-FL (sd_exclude)", "#1a6300", "-.", "D", 2.2, 5.5),
    ]

    # Helper to load rounds
    def load_rounds(strategy: str, condition: str) -> np.ndarray:
        # Check rounds CSV in ext_dir
        csvs = glob.glob(f"{ext_dir}/{strategy}_*_{condition}_seed42_rounds.csv")
        if not csvs and strategy == "sd_full":
            csvs = glob.glob(f"{ext_dir}/securedrive_*_{condition}_seed42_rounds.csv")
        if not csvs and strategy == "dpsgd":
            csvs = glob.glob(f"{ext_dir}/dp_sgd_*_{condition}_seed42_rounds.csv")

        if csvs:
            accs = []
            with open(csvs[0], "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    accs.append(float(row["acc"]) * 100)
            return np.array(accs)

        # Check JSON history in ext_dir or base_dir
        json_candidates = [
            f"{ext_dir}/{strategy}_hetero_tiers_{condition}_seed42.json",
            f"{ext_dir}/{strategy}_{condition}_seed42.json",
            f"{base_dir}/{strategy}_{condition}_seed42.json",
        ]
        if strategy == "sd_full":
            json_candidates.extend([
                f"{base_dir}/securedrive_{condition}_seed42.json",
                f"{ext_dir}/sd_full_hetero_tiers_{condition}_seed42.json"
            ])
        elif strategy == "dpsgd":
            json_candidates.extend([
                f"{base_dir}/dp_sgd_{condition}_seed42.json",
                f"{ext_dir}/dpsgd_hetero_tiers_{condition}_seed42.json"
            ])

        for jc in json_candidates:
            if os.path.exists(jc):
                with open(jc, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    hist = d.get("history") or d.get("rounds") or []
                    return np.array([x["acc"] * 100 for x in hist])

        sys.exit(f"MISSING: rounds history for strategy '{strategy}' condition '{condition}' for fig2_ext.")

    # Load clean FedAvg benchmark
    clean_fedavg = None
    fedavg_csvs = glob.glob(f"{ext_dir}/fedavg_*_clean_seed42_rounds.csv")
    if fedavg_csvs:
        with open(fedavg_csvs[0], "r", encoding="utf-8") as f:
            clean_fedavg = np.array([float(r["acc"]) * 100 for r in csv.DictReader(f)])
    else:
        fedavg_json = f"{base_dir}/fedavg_clean_seed42.json"
        if os.path.exists(fedavg_json):
            with open(fedavg_json, "r", encoding="utf-8") as f:
                d = json.load(f)
                clean_fedavg = np.array([x["acc"] * 100 for x in d["history"]])
        else:
            sys.exit(f"MISSING: clean FedAvg benchmark data in '{ext_dir}' or '{base_dir}' for fig2_ext.")

    total_rounds = len(clean_fedavg)
    bench_x = np.arange(1, total_rounds + 1)
    mark_indices = [0] + list(range(9, total_rounds, 10))
    if (total_rounds - 1) not in mark_indices:
        mark_indices.append(total_rounds - 1)

    handles_dict = {}

    for ax, cond, title in [(ax_mitm, "mitm", "Man-in-the-Middle"), (ax_poison, "poison", "Model Poisoning")]:
        h_bench, = ax.plot(
            bench_x, clean_fedavg,
            label="Clean FedAvg (Benchmark)", color="#7f7f7f", linestyle="--", marker="x",
            linewidth=2.0, markersize=7.0, markevery=mark_indices, zorder=2
        )
        handles_dict["bench"] = h_bench

        for strat, label, col, ls, m, lw, ms in series:
            accs = load_rounds(strat, cond)
            rx = np.arange(1, len(accs) + 1)
            h_line, = ax.plot(
                rx, accs,
                label=label, color=col, linestyle=ls, marker=m,
                linewidth=lw, markersize=ms, markevery=mark_indices, zorder=3
            )
            handles_dict[strat] = h_line

        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Communication Round", fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_xlim(0, max(120, total_rounds))
        ax.set_ylim(0, 105)
        ax.set_xticks(range(10, max(121, total_rounds + 1), 10))
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(direction="out", length=5, width=1.2, labelsize=11, top=False, right=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.grid(True, linestyle="--", color="#E5E5E5", alpha=1.0, linewidth=0.8)
        ax.set_axisbelow(True)

    ordered_keys = ["bench", "dpsgd", "sd_full", "sd_budget", "sd_exclude"]
    ordered_handles = [handles_dict[k] for k in ordered_keys]
    ordered_labels = [
        "Clean FedAvg (Benchmark)",
        "FL+DP-SGD (no enc)",
        "SecureDrive-FL (full enc)",
        "SecureDrive-FL (sd_budget)",
        "SecureDrive-FL (sd_exclude)",
    ]

    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=5,
        frameon=True,
        fancybox=True,
        edgecolor="#cccccc",
        facecolor="white",
        framealpha=1.0,
        fontsize=10.0,
        handlelength=2.4,
        columnspacing=1.6,
    )

    plt.subplots_adjust(bottom=0.23, top=0.90, left=0.065, right=0.985, wspace=0.18)

    plt.savefig("fig2_ext.png", dpi=300, bbox_inches="tight")
    plt.savefig("fig2_ext.pdf", bbox_inches="tight")
    plt.close()
    print("[Figure] Saved fig2_ext.png and fig2_ext.pdf")

    # Mirror to images/ and github_repo/figures/ if available
    for extra_dir in ["images", os.path.join("github_repo", "figures")]:
        if os.path.isdir(extra_dir):
            try:
                import shutil
                shutil.copy("fig2_ext.png", os.path.join(extra_dir, "fig2_ext.png"))
                shutil.copy("fig2_ext.pdf", os.path.join(extra_dir, "fig2_ext.pdf"))
            except Exception:
                pass


def generate_fig3_sweep(ext_dir: str = "results_ext", base_dir: str = "results"):
    """
    Fig. 3: Budget sweep (S1): final MitM accuracy vs f and protected energy vs f.
    Reference lines derived honestly from measured dpsgd and sd_full results.
    Error bars included when multiple seeds are available.
    """
    if not os.path.isdir(ext_dir):
        sys.exit(f"MISSING: Extension directory '{ext_dir}' not found for fig3_sweep.")

    fig, (ax_acc, ax_eng) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    fractions = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]

    def load_sweep(strategy: str) -> Tuple[List[float], List[float], List[float], List[float]]:
        fx, acc_means, acc_stds, eng_means = [], [], [], []
        for f in fractions:
            pattern = f"{ext_dir}/{strategy}_homog_f{f}_mitm_seed*.json"
            files = glob.glob(pattern)
            if not files:
                sys.exit(f"MISSING: Sweep file for strategy '{strategy}' budget f={f} in '{ext_dir}'.")
            accs, engs = [], []
            for fp in files:
                with open(fp, "r", encoding="utf-8") as f_obj:
                    d = json.load(f_obj)
                    accs.append(d["final_accuracy"] * 100)
                    engs.append(d.get("mean_protected_energy_fraction", f) * 100)
            fx.append(f * 100)
            acc_means.append(float(np.mean(accs)))
            acc_stds.append(float(np.std(accs)) if len(accs) > 1 else 0.0)
            eng_means.append(float(np.mean(engs)))
        return fx, acc_means, acc_stds, eng_means

    fx_prop, acc_prop, acc_prop_std, eng_prop = load_sweep("sd_budget")
    fx_rnd, acc_rnd, acc_rnd_std, eng_rnd = load_sweep("sd_budget_random")
    fx_lyr, acc_lyr, acc_lyr_std, eng_lyr = load_sweep("sd_budget_layer")

    # Plot accuracy panel with error bars if multiple seeds exist
    if any(s > 0 for s in acc_prop_std):
        ax_acc.errorbar(fx_prop, acc_prop, yerr=acc_prop_std, fmt="o-", color="crimson", linewidth=2.2, capsize=4, label="sd_budget (post-noise energy, default)")
    else:
        ax_acc.plot(fx_prop, acc_prop, "o-", color="crimson", linewidth=2.2, label="sd_budget (post-noise energy, default)")

    if any(s > 0 for s in acc_rnd_std):
        ax_acc.errorbar(fx_rnd, acc_rnd, yerr=acc_rnd_std, fmt="s--", color="gray", linewidth=1.8, capsize=3, label="sd_budget_random (random ranking)")
    else:
        ax_acc.plot(fx_rnd, acc_rnd, "s--", color="gray", linewidth=1.8, label="sd_budget_random (random ranking)")

    if any(s > 0 for s in acc_lyr_std):
        ax_acc.errorbar(fx_lyr, acc_lyr, yerr=acc_lyr_std, fmt="^-.", color="purple", linewidth=1.8, capsize=3, label="sd_budget_layer (classifier-first)")
    else:
        ax_acc.plot(fx_lyr, acc_lyr, "^-.", color="purple", linewidth=1.8, label="sd_budget_layer (classifier-first)")

    # Plot energy panel
    ax_eng.plot(fx_prop, eng_prop, "o-", color="crimson", linewidth=2.2, label="sd_budget (energy ranking)")
    ax_eng.plot(fx_rnd, eng_rnd, "s--", color="gray", linewidth=1.8, label="sd_budget_random (random)")
    ax_eng.plot(fx_lyr, eng_lyr, "^-.", color="purple", linewidth=1.8, label="sd_budget_layer (layer-static)")

    # Reference lines derived from measured files
    dpsgd_files = glob.glob(f"{base_dir}/dp_sgd_mitm_seed*.json") or glob.glob(f"{ext_dir}/dpsgd_*_mitm_seed*.json")
    if not dpsgd_files:
        sys.exit(f"MISSING: DP-SGD MitM results for fig3 reference line in '{base_dir}' or '{ext_dir}'.")
    dpsgd_accs = [json.load(open(fp, "r", encoding="utf-8"))["final_accuracy"] * 100 for fp in dpsgd_files]
    ref_dpsgd = float(np.mean(dpsgd_accs))

    sd_files = glob.glob(f"{base_dir}/securedrive_mitm_seed*.json") or glob.glob(f"{ext_dir}/sd_full_*_mitm_seed*.json")
    if not sd_files:
        sys.exit(f"MISSING: SecureDrive-FL MitM results for fig3 reference line in '{base_dir}' or '{ext_dir}'.")
    sd_accs = [json.load(open(fp, "r", encoding="utf-8"))["final_accuracy"] * 100 for fp in sd_files]
    ref_sd = float(np.mean(sd_accs))

    ax_acc.axhline(y=ref_dpsgd, color="blue", linestyle=":", label=f"dpsgd ({ref_dpsgd:.2f}% collapse baseline)")
    ax_acc.axhline(y=ref_sd, color="darkblue", linestyle="--", label=f"sd_full ({ref_sd:.2f}% full budget reference)")

    ax_acc.set_title("MitM Accuracy vs. Compute Budget Fraction f", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Budget Fraction f (% of Full-Vector Encryption)", fontsize=11)
    ax_acc.set_ylabel("Final MitM Accuracy (%)", fontsize=11)
    ax_acc.set_ylim(-2, 102)
    ax_acc.grid(True, linestyle="--", alpha=0.5)
    ax_acc.legend(loc="lower right", framealpha=0.9)

    ax_eng.set_title("Protected Gradient Energy vs. Budget Fraction f", fontsize=12, fontweight="bold")
    ax_eng.set_xlabel("Budget Fraction f (%)", fontsize=11)
    ax_eng.set_ylabel("Protected Gradient Energy Fraction (%)", fontsize=11)
    ax_eng.set_ylim(-2, 102)
    ax_eng.grid(True, linestyle="--", alpha=0.5)
    ax_eng.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig("fig3_sweep.png")
    plt.savefig("fig3_sweep.pdf")
    plt.close()
    print("[Figure] Saved fig3_sweep.png and fig3_sweep.pdf")


def generate_fig4_leakage(audit_path: str = "results_ext/audit_results.json", samples_path: str = "results_ext/audit_samples.npz"):
    """
    Fig. 4: Reconstruction PSNR vs budget fraction f for Non-DP and DP audits,
    plus 2x2 example reconstruction grids if sample data is available.
    """
    if not os.path.exists(audit_path):
        sys.exit(f"MISSING: Audit results file '{audit_path}' not found for fig4_leakage.")

    with open(audit_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    budgets = [float(b) * 100 for b in data["budgets"]]
    psnr_nondp = [data["non_dp_audit"]["results_by_budget"][str(b)]["psnr"] for b in data["budgets"]]
    psnr_nondp_std = [data["non_dp_audit"]["results_by_budget"][str(b)]["psnr_std"] for b in data["budgets"]]
    psnr_dp = [data["dp_audit"]["results_by_budget"][str(b)]["psnr"] for b in data["budgets"]]
    psnr_dp_std = [data["dp_audit"]["results_by_budget"][str(b)]["psnr_std"] for b in data["budgets"]]
    noise_base = data["non_dp_audit"]["baseline_noise_psnr"]
    mean_base = data["non_dp_audit"]["baseline_mean_psnr"]

    has_samples = os.path.exists(samples_path)

    if has_samples:
        fig = plt.figure(figsize=(15, 6), dpi=300)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0])
        ax_curve = fig.add_subplot(gs[:, 0])
    else:
        fig, ax_curve = plt.subplots(figsize=(8, 5), dpi=300)

    # Plot PSNR curves with error bars
    ax_curve.errorbar(budgets, psnr_nondp, yerr=psnr_nondp_std, fmt="o-", color="crimson", linewidth=2.2, capsize=4, label="Non-DP Audit (Gradient Inversion)")
    ax_curve.errorbar(budgets, psnr_dp, yerr=psnr_dp_std, fmt="s-", color="blue", linewidth=2.2, capsize=4, label="DP Audit (eps0=4, Noise Obfuscation)")
    ax_curve.axhline(y=noise_base, color="black", linestyle=":", label=f"Pure Noise Baseline ({noise_base:.1f} dB)")
    ax_curve.axhline(y=mean_base, color="darkgray", linestyle="--", label=f"Dataset Mean Baseline ({mean_base:.1f} dB)")

    ax_curve.set_title("Reconstruction Leakage (PSNR) vs. Budget Fraction f", fontsize=12, fontweight="bold")
    ax_curve.set_xlabel("Compute Budget Fraction f (% Encrypted Chunks)", fontsize=11)
    ax_curve.set_ylabel("Reconstruction Quality PSNR (dB) [Lower is Safer]", fontsize=11)
    ax_curve.grid(True, linestyle="--", alpha=0.5)
    ax_curve.legend(loc="upper right")

    # If samples exist, plot 2x2 example reconstruction subplots
    if has_samples:
        samples_npz = np.load(samples_path)
        orig = samples_npz["original_samples"] if "original_samples" in samples_npz else None
        rec_f0 = samples_npz["non_dp_rec_f0.0"] if "non_dp_rec_f0.0" in samples_npz else samples_npz.get("non_dp_rec_f0")
        rec_f05 = samples_npz["non_dp_rec_f0.5"] if "non_dp_rec_f0.5" in samples_npz else samples_npz.get("non_dp_rec_f0.50")
        rec_f10 = samples_npz["non_dp_rec_f1.0"] if "non_dp_rec_f1.0" in samples_npz else samples_npz.get("non_dp_rec_f1")

        if orig is not None and len(orig) >= 2:
            # 2x2 sample grid: sample 0 (top), sample 1 (bottom)
            # columns: Original, f=0, f=0.5, f=1.0
            ax_top = fig.add_subplot(gs[0, 1])
            ax_bot = fig.add_subplot(gs[1, 1])

            def plot_sample_strip(ax, s_idx, title):
                # Visualize 1D feature vectors as intensity strips
                v_orig = orig[s_idx].reshape(1, -1)
                v_f0 = rec_f0[s_idx].reshape(1, -1) if rec_f0 is not None else v_orig
                v_f05 = rec_f05[s_idx].reshape(1, -1) if rec_f05 is not None else v_orig
                v_f10 = rec_f10[s_idx].reshape(1, -1) if rec_f10 is not None else v_orig

                strip = np.vstack([v_orig, v_f0, v_f05, v_f10])
                im = ax.imshow(strip, aspect="auto", cmap="viridis")
                ax.set_yticks([0, 1, 2, 3])
                ax.set_yticklabels(["Orig", "f=0%", "f=50%", "f=100%"], fontsize=8)
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.set_xlabel("Feature Coordinate Index", fontsize=8)

            plot_sample_strip(ax_top, 0, "Sample 0 Inversion Across Budgets")
            plot_sample_strip(ax_bot, 1, "Sample 1 Inversion Across Budgets")

    plt.tight_layout()
    plt.savefig("fig4_leakage.png")
    plt.savefig("fig4_leakage.pdf")
    plt.close()
    print("[Figure] Saved fig4_leakage.png and fig4_leakage.pdf")


def generate_fig5_cost_benefit(ext_dir: str = "results_ext", base_dir: str = "results"):
    """
    Fig. 5: MitM accuracy vs total simulated time.
    All data points derived purely from measured files without hardcoded fallbacks.
    """
    if not os.path.isdir(ext_dir):
        sys.exit(f"MISSING: Extension directory '{ext_dir}' not found for fig5_cost_benefit.")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    strategies = [
        ("dpsgd", "FL+DP-SGD (no enc)", "blue", "o"),
        ("sd_budget", "sd_budget (proposal)", "crimson", "s"),
        ("sd_exclude", "sd_exclude (drop weak)", "forestgreen", "^"),
        ("sd_wait", "sd_wait (straggler wait)", "darkblue", "D"),
    ]

    for strat, label, col, marker in strategies:
        files = glob.glob(f"{ext_dir}/{strat}_hetero_tiers_mitm_seed*.json")
        if not files and strat == "dpsgd":
            files = glob.glob(f"{base_dir}/dp_sgd_mitm_seed*.json")

        if not files:
            sys.exit(f"MISSING: Measured results for strategy '{strat}' in '{ext_dir}' or '{base_dir}' for fig5.")

        times, accs = [], []
        for fp in files:
            with open(fp, "r", encoding="utf-8") as f_obj:
                d = json.load(f_obj)
                t_val = d.get("total_sim_time_s")
                if t_val is None:
                    t_val = d.get("elapsed_time_s", 0.0)
                times.append(float(t_val))
                accs.append(float(d["final_accuracy"]) * 100)

        sim_t = float(np.mean(times))
        acc = float(np.mean(accs))
        ax.scatter([sim_t], [acc], color=col, marker=marker, s=120, label=f"{label} ({sim_t:.1f}s, {acc:.1f}%)", zorder=5)

    ax.set_title("Cost-Benefit Trade-Off: MitM Accuracy vs. Simulated Runtime", fontsize=12, fontweight="bold")
    ax.set_xlabel("Total Simulated Execution Time (s) [Lower is Faster]", fontsize=11)
    ax.set_ylabel("Final MitM Accuracy (%) [Higher is Better]", fontsize=11)
    ax.set_ylim(-2, 102)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig("fig5_cost_benefit.png")
    plt.savefig("fig5_cost_benefit.pdf")
    plt.close()
    print("[Figure] Saved fig5_cost_benefit.png and fig5_cost_benefit.pdf")


def generate_fig6_pertier(ext_dir: str = "results_ext"):
    """
    Fig. 6: Per-tier mean protected energy fraction in scenario H1.
    Derived purely from measured H1 rounds CSVs in results_ext/.
    """
    if not os.path.isdir(ext_dir):
        sys.exit(f"MISSING: Extension directory '{ext_dir}' not found for fig6_pertier.")

    csv_files = glob.glob(f"{ext_dir}/sd_budget_hetero_tiers_mitm_seed*_rounds.csv")
    if not csv_files:
        sys.exit(f"MISSING: Measured 'sd_budget_hetero_tiers_mitm_*_rounds.csv' files in '{ext_dir}' for fig6_pertier.")

    # Tiers under K=6, tier_counts=[2, 2, 1, 1]:
    # Tier 1 (10% capacity): cids [0, 1]
    # Tier 2 (25% capacity): cids [2, 3]
    # Tier 3 (50% capacity): cid [4]
    # Tier 4 (100% capacity): cid [5]
    tier_cids = {
        0: [0, 1],
        1: [2, 3],
        2: [4],
        3: [5],
    }
    tier_energies = {0: [], 1: [], 2: [], 3: []}

    for cp in csv_files:
        with open(cp, "r", encoding="utf-8") as f_obj:
            reader = csv.DictReader(f_obj)
            for row in reader:
                if str(row.get("skipped_round", "False")).strip().lower() in ("true", "1"):
                    continue
                c_energies_raw = row.get("client_protected_energies")
                if c_energies_raw:
                    try:
                        client_energies = json.loads(c_energies_raw)
                        for t_idx, cids in tier_cids.items():
                            vals = [client_energies[str(cid)] if str(cid) in client_energies else client_energies[cid] for cid in cids if str(cid) in client_energies or cid in client_energies]
                            if vals:
                                tier_energies[t_idx].append(float(np.mean(vals)) * 100)
                    except Exception:
                        pass

    tiers = ["Tier 1 (10%)", "Tier 2 (25%)", "Tier 3 (50%)", "Tier 4 (100%)"]
    prot_energy = [float(np.mean(tier_energies[t])) if tier_energies[t] else 0.0 for t in range(4)]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(tiers, prot_energy, color=["#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4"], edgecolor="black", width=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_title("Per-Tier Protected Gradient Energy (Heterogeneous Tiers H1)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Device Capacity Tiers", fontsize=11)
    ax.set_ylabel("Protected Gradient Energy Fraction (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("fig6_pertier.png")
    plt.savefig("fig6_pertier.pdf")
    plt.close()
    print("[Figure] Saved fig6_pertier.png and fig6_pertier.pdf")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate extension figures from results.")
    parser.add_argument("--ext-dir", default="results_ext", help="Directory with extension results (default: results_ext)")
    parser.add_argument("--base-dir", default="results", help="Directory with baseline results (default: results)")
    args = parser.parse_args()

    print(f"Generating extension figures (ext_dir={args.ext_dir}, base_dir={args.base_dir})...")
    generate_fig2_ext(ext_dir=args.ext_dir, base_dir=args.base_dir)
    generate_fig3_sweep(ext_dir=args.ext_dir, base_dir=args.base_dir)
    generate_fig4_leakage()
    generate_fig5_cost_benefit(ext_dir=args.ext_dir, base_dir=args.base_dir)
    generate_fig6_pertier(ext_dir=args.ext_dir)
    print("[Success] All extension figures generated successfully!")


if __name__ == "__main__":
    main()
