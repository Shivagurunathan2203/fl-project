#!/usr/bin/env python3
"""
make_figure_computation_cost.py
===============================
Generates publication-quality figures comparing computation cost vs number of rounds
for FedAvg, Base Paper (SecureDrive-FL), and all extension models.

Outputs:
  - fig_computation_cost.png / .pdf         : Dual-panel (Full Scale & Zoomed Budgeted View)
  - fig_computation_cost_single.png / .pdf  : Single-panel standalone figure
All styled matching the original paper's aesthetic with external bottom legend.
"""

import os
import sys
import glob
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def setup_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })


def load_model_data(ext_dir: str = "results_ext", base_dir: str = "results"):
    def get_ext_times(key: str):
        csvs = glob.glob(f"{ext_dir}/{key}_hetero_tiers_mitm_seed*_rounds.csv")
        if not csvs:
            csvs = glob.glob(f"{ext_dir}/{key}_*mitm_seed*_rounds.csv")
        if csvs:
            all_times = []
            for c in csvs:
                with open(c, "r", encoding="utf-8") as fp:
                    r = list(csv.DictReader(fp))
                    all_times.append([float(x["sim_round_time_s"]) for x in r])
            mean_times = np.mean(all_times, axis=0)
            return np.cumsum(mean_times), mean_times
        return None, None

    cum_wait, r_wait = get_ext_times("sd_wait")
    cum_budget, r_budget = get_ext_times("sd_budget")
    cum_exclude, r_exclude = get_ext_times("sd_exclude")
    cum_random, r_random = get_ext_times("sd_budget_random")
    cum_layer, r_layer = get_ext_times("sd_budget_layer")
    cum_gashe, r_gashe = get_ext_times("gashe_budget")

    rounds = np.arange(1, 121)

    # FedAvg plaintext baseline accumulation from calibrated timing (Table II)
    t_fedavg_per_rd = 11.2 / 120.0
    cum_fedavg = rounds * t_fedavg_per_rd

    # Base paper sd_full (ideal homogeneous budget) accumulation (Table II)
    t_full_per_rd = 32.4 / 120.0
    cum_full = rounds * t_full_per_rd

    models_data = [
        ("fedavg", "FedAvg (Plaintext Baseline)", "#7f7f7f", ":", "x", 2.0, 7.0, cum_fedavg),
        ("sd_full", "Base Paper (sd_full, ideal)", "#2ca02c", "--", "d", 2.2, 5.5, cum_full),
        ("sd_wait", "Base Paper (sd_wait, straggler wait)", "#08306b", "-", "o", 2.5, 6.0, cum_wait),
        ("sd_budget", "Our Proposal (sd_budget)", "#a2142f", "-", "P", 2.5, 7.0, cum_budget),
        ("sd_exclude", "Client Exclusion (sd_exclude)", "#e66101", "-.", "s", 2.2, 5.5, cum_exclude),
        ("sd_random", "Random Selection (sd_budget_random)", "#762a83", "-.", "v", 2.2, 6.0, cum_random),
        ("sd_layer", "Layer-Static (sd_budget_layer)", "#1f78b4", "--", "^", 2.2, 6.0, cum_layer),
        ("gashe_budget", "Non-DP Budgeted (gashe_budget)", "#005a00", ":", "D", 2.2, 5.5, cum_gashe),
    ]
    return rounds, models_data


def generate_figures(ext_dir: str = "results_ext", base_dir: str = "results", out_dir: str = "."):
    setup_paper_style()
    rounds, models_data = load_model_data(ext_dir=ext_dir, base_dir=base_dir)
    mark_indices = [0] + list(range(9, 120, 10))

    # =========================================================================
    # 1. Dual-Panel Figure: Full-Scale + Zoomed View
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=300)
    handles, labels = [], []

    for key, lbl, col, ls, m, lw, ms, cum_t in models_data:
        h1, = ax1.plot(
            rounds, cum_t, label=lbl, color=col, linestyle=ls, marker=m,
            linewidth=lw, markersize=ms, markevery=mark_indices, zorder=3
        )
        handles.append(h1)
        labels.append(lbl)

        ax2.plot(
            rounds, cum_t, label=lbl, color=col, linestyle=ls, marker=m,
            linewidth=lw, markersize=ms, markevery=mark_indices, zorder=3
        )

    # Left Panel (Full Scale)
    ax1.set_title("Cumulative Computation Cost (Full Scale)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Communication Round", fontsize=12)
    ax1.set_ylabel("Cumulative Computation Time (s)", fontsize=12)
    ax1.set_xlim(0, 120)
    ax1.set_ylim(0, 360)
    ax1.set_xticks(range(10, 121, 10))
    ax1.set_yticks(range(0, 361, 50))
    ax1.tick_params(direction="out", length=5, width=1.2, labelsize=11, top=False, right=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(1.2)
    ax1.spines["bottom"].set_linewidth(1.2)
    ax1.grid(True, linestyle="--", color="#E5E5E5", alpha=1.0, linewidth=0.8)
    ax1.set_axisbelow(True)

    # Right Panel (Zoomed View 0-40s)
    ax2.set_title("Zoomed View: Budgeted & Baseline Models", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Communication Round", fontsize=12)
    ax2.set_ylabel("Cumulative Computation Time (s)", fontsize=12)
    ax2.set_xlim(0, 120)
    ax2.set_ylim(0, 40)
    ax2.set_xticks(range(10, 121, 10))
    ax2.set_yticks(range(0, 41, 5))
    ax2.tick_params(direction="out", length=5, width=1.2, labelsize=11, top=False, right=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(1.2)
    ax2.spines["bottom"].set_linewidth(1.2)
    ax2.grid(True, linestyle="--", color="#E5E5E5", alpha=1.0, linewidth=0.8)
    ax2.set_axisbelow(True)

    # Shared External Legend at Bottom
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=4,
        frameon=True,
        fancybox=True,
        edgecolor="#cccccc",
        facecolor="white",
        framealpha=1.0,
        fontsize=9.8,
        handlelength=2.4,
        columnspacing=1.5,
    )

    plt.subplots_adjust(bottom=0.26, top=0.90, left=0.07, right=0.98, wspace=0.20)

    p1_png = os.path.join(out_dir, "fig_computation_cost.png")
    p1_pdf = os.path.join(out_dir, "fig_computation_cost.pdf")
    plt.savefig(p1_png, dpi=300, bbox_inches="tight")
    plt.savefig(p1_pdf, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {p1_png} and {p1_pdf}")

    # =========================================================================
    # 2. Single-Panel Standalone Figure
    # =========================================================================
    fig_s, ax_s = plt.subplots(figsize=(8.0, 5.2), dpi=300)
    handles_s, labels_s = [], []

    for key, lbl, col, ls, m, lw, ms, cum_t in models_data:
        h, = ax_s.plot(
            rounds, cum_t, label=lbl, color=col, linestyle=ls, marker=m,
            linewidth=lw, markersize=ms, markevery=mark_indices, zorder=3
        )
        handles_s.append(h)
        labels_s.append(lbl)

    ax_s.set_title("Computation Cost vs. Number of Communication Rounds", fontsize=13, fontweight="bold", pad=10)
    ax_s.set_xlabel("Communication Round", fontsize=12)
    ax_s.set_ylabel("Cumulative Computation Time (s)", fontsize=12)
    ax_s.set_xlim(0, 120)
    ax_s.set_ylim(0, 360)
    ax_s.set_xticks(range(10, 121, 10))
    ax_s.tick_params(direction="out", length=5, width=1.2, labelsize=11, top=False, right=False)
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    ax_s.spines["left"].set_linewidth(1.2)
    ax_s.spines["bottom"].set_linewidth(1.2)
    ax_s.grid(True, linestyle="--", color="#E5E5E5", alpha=1.0, linewidth=0.8)
    ax_s.set_axisbelow(True)

    fig_s.legend(
        handles_s,
        labels_s,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True,
        fancybox=True,
        edgecolor="#cccccc",
        facecolor="white",
        framealpha=1.0,
        fontsize=9.2,
        handlelength=2.2,
        columnspacing=1.2,
    )

    plt.subplots_adjust(bottom=0.28, top=0.92, left=0.10, right=0.96)
    p2_png = os.path.join(out_dir, "fig_computation_cost_single.png")
    p2_pdf = os.path.join(out_dir, "fig_computation_cost_single.pdf")
    plt.savefig(p2_png, dpi=300, bbox_inches="tight")
    plt.savefig(p2_pdf, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {p2_png} and {p2_pdf}")

    # Mirror copies to images/ and github_repo/figures/
    for extra_dir in ["images", os.path.join("github_repo", "figures")]:
        dest_dir = os.path.join(out_dir, extra_dir) if out_dir != "." else extra_dir
        if os.path.isdir(dest_dir):
            for fn in ["fig_computation_cost.png", "fig_computation_cost.pdf",
                       "fig_computation_cost_single.png", "fig_computation_cost_single.pdf"]:
                src = os.path.join(out_dir, fn)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(dest_dir, fn))
            print(f"[Mirrored] Figures copied to {dest_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate computation cost vs rounds figures.")
    parser.add_argument("--ext-dir", default="results_ext", help="Directory with extension results (default: results_ext)")
    parser.add_argument("--base-dir", default="results", help="Directory with baseline results (default: results)")
    parser.add_argument("--out-dir", default=".", help="Output directory (default: .)")
    args = parser.parse_args()

    generate_figures(ext_dir=args.ext_dir, base_dir=args.base_dir, out_dir=args.out_dir)
