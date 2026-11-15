#!/usr/bin/env python3
"""
Diagnostic & Analysis Plot: Root Causes of MitM 10% Accuracy.
Generates a publication-quality 3-panel figure:
  Panel A: Convergence Trajectories under Calibrated Benchmark (MitM vs Clean)
  Panel B: The Diagnostic of Table 1 (Held-out vs synthetic dynamics)
  Panel C: Final Clean vs. MitM Accuracy Comparison across Key FL Paradigms
Strictly faithful to final_spec.txt:
- Panel C values loaded purely from measured results in results/.
- Exits non-zero if results are missing.
"""

import os
import sys
import glob
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from securedrive_fl.config import Config
from securedrive_fl.data import make_toy_federation
from securedrive_fl.he import MockHE
from securedrive_fl.run_scenario import run_scenario


def run_calibrated_trajectories(rounds: int = 25, seed: int = 42):
    """Runs short calibrated runs to get clean and MitM trajectory curves."""
    print("Running calibrated scenarios for Panel A...")
    client_data, test_data, K = make_toy_federation(seed=seed)
    he = MockHE()

    # 1. Clean FedAvg
    cfg_clean_fedavg = Config(rounds=rounds, seed=seed, use_dp=False, use_gashe=False, attack=None)
    hist_clean_fedavg = run_scenario(cfg_clean_fedavg, client_data, test_data, K, he)

    # 2. Clean SecureDrive-FL (DP-SGD)
    cfg_clean_sd = Config(rounds=rounds, seed=seed, use_dp=True, use_gashe=True, attack=None)
    hist_clean_sd = run_scenario(cfg_clean_sd, client_data, test_data, K, he)

    # 3. SecureDrive-FL under MitM
    cfg_mitm_sd = Config(rounds=rounds, seed=seed, use_dp=True, use_gashe=True, attack="mitm", loss_guard=True)
    hist_mitm_sd = run_scenario(cfg_mitm_sd, client_data, test_data, K, he)

    # 4. FedAvg under MitM
    cfg_mitm_fedavg = Config(rounds=rounds, seed=seed, use_dp=False, use_gashe=False, attack="mitm")
    hist_mitm_fedavg = run_scenario(cfg_mitm_fedavg, client_data, test_data, K, he)

    # 5. DP-SGD under MitM
    cfg_mitm_dpsgd = Config(rounds=rounds, seed=seed, use_dp=True, use_gashe=False, attack="mitm")
    hist_mitm_dpsgd = run_scenario(cfg_mitm_dpsgd, client_data, test_data, K, he)

    return {
        "clean_fedavg": [h["acc"] * 100 for h in hist_clean_fedavg],
        "clean_sd": [h["acc"] * 100 for h in hist_clean_sd],
        "mitm_sd": [h["acc"] * 100 for h in hist_mitm_sd],
        "mitm_fedavg": [h["acc"] * 100 for h in hist_mitm_fedavg],
        "mitm_dpsgd": [h["acc"] * 100 for h in hist_mitm_dpsgd],
    }


def load_table1_raw_results(results_dir: str = "results"):
    """Loads actual history from results/*.json files that formed Table 1."""
    def load_hist(fname):
        p = Path(results_dir) / fname
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
                return [h["acc"] * 100 for h in d["history"]]
        return None

    return {
        "fedavg_clean": load_hist("fedavg_clean_seed42.json"),
        "securedrive_clean": load_hist("securedrive_clean_seed42.json"),
        "fedavg_mitm": load_hist("fedavg_mitm_seed42.json"),
        "securedrive_mitm": load_hist("securedrive_mitm_seed42.json"),
        "dp_sgd_mitm": load_hist("dp_sgd_mitm_seed42.json"),
    }


def load_panel_c_data(results_dir: str = "results"):
    """Loads measured Clean and MitM accuracies for Panel C from results/."""
    method_keys = [
        ("FedAvg", "fedavg"),
        ("FedProx", "fedprox"),
        ("DP-SGD\n(eps0=4)", "dp_sgd"),
        ("GASHE\nonly", "gashe_only"),
        ("SecureDrive\n-FL", "securedrive"),
        ("Trimmed\nMean", "trimmed_mean"),
        ("Krum", "krum"),
    ]

    labels = []
    clean_accs = []
    mitm_accs = []

    for display_label, prefix in method_keys:
        clean_files = glob.glob(f"{results_dir}/{prefix}_clean_seed*.json")
        mitm_files = glob.glob(f"{results_dir}/{prefix}_mitm_seed*.json")

        if not clean_files:
            sys.exit(f"MISSING: {results_dir}/{prefix}_clean_seed*.json for plot_mitm_analysis Panel C.")
        if not mitm_files:
            sys.exit(f"MISSING: {results_dir}/{prefix}_mitm_seed*.json for plot_mitm_analysis Panel C.")

        c_vals = [json.load(open(fp, "r", encoding="utf-8"))["final_accuracy"] * 100 for fp in clean_files]
        m_vals = [json.load(open(fp, "r", encoding="utf-8"))["final_accuracy"] * 100 for fp in mitm_files]

        labels.append(display_label)
        clean_accs.append(float(np.mean(c_vals)))
        mitm_accs.append(float(np.mean(m_vals)))

    return labels, clean_accs, mitm_accs


def plot_all(results_dir: str = "results"):
    if not os.path.isdir(results_dir):
        sys.exit(f"MISSING: Results directory '{results_dir}' does not exist.")

    calib = run_calibrated_trajectories(rounds=20)
    raw = load_table1_raw_results(results_dir)
    methods, clean_accs, mitm_accs = load_panel_c_data(results_dir)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.8), dpi=300)

    # -------------------------------------------------------------
    # Panel A: Calibrated MitM Dynamics (Why Unprotected Fails to ~10%)
    # -------------------------------------------------------------
    rx = np.arange(1, len(calib["clean_fedavg"]) + 1)
    ax1.plot(rx, calib["clean_fedavg"], label="Clean FedAvg (Baseline)", color="#2b5c8f", linestyle="--", linewidth=2.0)
    ax1.plot(rx, calib["clean_sd"], label="Clean SecureDrive (DP-SGD)", color="#17becf", linestyle="--", linewidth=1.8)
    ax1.plot(rx, calib["mitm_sd"], label="SecureDrive-FL (MitM + Guard)", color="#2ca02c", linestyle="-", linewidth=2.5)
    ax1.plot(rx, calib["mitm_dpsgd"], label="FL + DP-SGD (MitM, Unprotected)", color="#ff7f0e", linestyle="-.", linewidth=2.0)
    ax1.plot(rx, calib["mitm_fedavg"], label="Standard FedAvg (MitM, Collapsed)", color="#d62728", linestyle="-", linewidth=2.2)

    ax1.axhline(10.0, color="#8b0000", linestyle=":", linewidth=2.0, alpha=0.9, label="10% Random Chance (1/K, K=10)")
    ax1.text(1.2, 11.5, "Random Guess Floor (~10%)", color="#8b0000", fontweight="bold", fontsize=9)

    ax1.set_title("A. MitM Attack Dynamics", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Communication Round", fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax1.set_ylim(-3, 103)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="center right", fontsize=8.5, framealpha=0.92)

    ax1.annotate(
        "MitM coordinate perturbation\ndiverges plaintext weights",
        xy=(2, 2.0), xytext=(4, 25),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
        fontsize=8.5, color="#d62728", fontweight="semibold"
    )

    # -------------------------------------------------------------
    # Panel B: Table 1 Diagnostic / Progression
    # -------------------------------------------------------------
    if raw["fedavg_clean"]:
        rx_raw = np.arange(1, len(raw["fedavg_clean"]) + 1)
        ax2.plot(rx_raw, raw["fedavg_clean"], label="FedAvg (Clean)", color="#2b5c8f", linestyle="--", linewidth=1.8)
        ax2.plot(rx_raw, raw["securedrive_clean"], label="SecureDrive-FL (Clean)", color="#17becf", linestyle="--", linewidth=1.8)
        ax2.plot(rx_raw, raw["securedrive_mitm"], label="SecureDrive-FL (MitM)", color="#2ca02c", linestyle="-", linewidth=2.0)
        ax2.plot(rx_raw, raw["dp_sgd_mitm"], label="DP-SGD (MitM)", color="#ff7f0e", linestyle="-.", linewidth=1.8)
        ax2.plot(rx_raw, raw["fedavg_mitm"], label="FedAvg (MitM)", color="#d62728", linestyle="-", linewidth=1.8)

    ax2.axhline(10.0, color="#8b0000", linestyle=":", linewidth=2.0, alpha=0.9, label="10% Random Floor")
    ax2.set_title("B. Measured Round Trajectories", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Communication Round", fontsize=11)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax2.set_ylim(-3, 103)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    bbox_props = dict(boxstyle="round,pad=0.5", fc="#fff3cd", ec="#856404", lw=1.2)
    sample_json = glob.glob(f"{results_dir}/*.json")[0]
    with open(sample_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    d_src = meta.get("data_source", "unknown")
    r_cnt = meta.get("rounds", len(rx_raw) if raw["fedavg_clean"] else 0)

    ax2.text(
        r_cnt * 0.5, 45,
        f"Data source: {d_src}\nRounds: {r_cnt}\n"
        f"Unprotected FL collapses\nunder MitM tampering;\n"
        f"GASHE encryption maintains\nstability and convergence.",
        fontsize=9, ha="center", va="center", bbox=bbox_props
    )

    # -------------------------------------------------------------
    # Panel C: Clean vs. MitM Across Paradigms (Measured Data)
    # -------------------------------------------------------------
    x = np.arange(len(methods))
    w = 0.38

    b1 = ax3.bar(x - w/2, clean_accs, w, label="Clean", color="#4a7ebb", edgecolor="black", linewidth=0.7)
    b2 = ax3.bar(x + w/2, mitm_accs, w, label="MitM Attack", color="#e15759", edgecolor="black", linewidth=0.7)

    # Value labels on top of bars
    for rects, color in [(b1, "#224466"), (b2, "#881111")]:
        for rect in rects:
            h = rect.get_height()
            ax3.annotate(f"{h:.1f}%",
                         xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold")

    ax3.axhline(10.0, color="#8b0000", linestyle=":", linewidth=2.0, alpha=0.9, label="10% Random Floor")
    ax3.set_title("C. Clean vs. MitM Resilience Comparison", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, fontsize=9.5)
    ax3.set_ylabel("Final Test Accuracy (%)", fontsize=11)
    ax3.set_ylim(-3, 108)
    ax3.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    out_png = PROJECT_ROOT / "fig_mitm_analysis.png"
    out_pdf = PROJECT_ROOT / "fig_mitm_analysis.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"[Success] Saved diagnostic plot to:\n  - {out_png}\n  - {out_pdf}")


if __name__ == "__main__":
    plot_all()
