from __future__ import annotations

import os
import sys
import glob
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt

CONFIG_DISPLAY = {
    "fedavg": ("Standard FL (FedAvg)", False),
    "dp_sgd": ("FL + DP-SGD (eps0=4)", True),
    "gashe_only": ("FL + GASHE-only", False),
    "full_ckks": ("FL + Full-CKKS", False),
    "securedrive": ("SecureDrive-FL", True),
    "trimmed_mean": ("TrimmedMean", False),
    "krum": ("Krum", False),
    "batchcrypt": ("BatchCrypt", False),
    "fedprox": ("FedProx", False),
}

ORDERED_CONFIGS = [
    "fedavg",
    "dp_sgd",
    "gashe_only",
    "full_ckks",
    "securedrive",
    "trimmed_mean",
    "krum",
    "batchcrypt",
    "fedprox",
]

SERIES_FIG2_INFO = {
    "bench": ("Benchmark", "#7f7f7f", "--", "x", 2.0, 7.0),
    "fedavg": ("(1) FedAvg", "#d65f4d", ":", "s", 2.2, 5.5),
    "gashe_only": ("(2) FL+GASHE", "#762a83", "-", "^", 2.4, 6.0),
    "batchcrypt": ("(3) BatchCrypt", "#a2142f", "-.", "v", 2.2, 6.0),
    "dp_sgd": (r"(4) FL+DP-SGD ($\varepsilon=4$)", "#4daf4a", "-", "D", 2.4, 5.5),
    "securedrive": (r"(5) SecureDrive-FL ($\varepsilon=4$)", "#005a00", "-", "P", 2.5, 7.0),
}
# Column-first order for matplotlib ncol=3 to yield:
# Col 1: Benchmark, (1) FedAvg
# Col 2: (2) FL+GASHE, (3) BatchCrypt
# Col 3: (4) FL+DP-SGD, (5) SecureDrive-FL
LEGEND_ORDER_FIG2 = ["bench", "fedavg", "gashe_only", "batchcrypt", "dp_sgd", "securedrive"]


def load_all_results(results_dir: str = "results") -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Loads results organized by config_name -> condition -> list of seed results."""
    if not os.path.isdir(results_dir):
        sys.exit(f"MISSING: Results directory '{results_dir}' does not exist.")

    files = glob.glob(os.path.join(results_dir, "*.json"))
    if not files:
        sys.exit(f"MISSING: No result JSON files found in '{results_dir}'.")

    data: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            res = json.load(fp)
        cfg = res["config_name"]
        cond = res["condition"]
        if cfg not in data:
            data[cfg] = {}
        if cond not in data[cfg]:
            data[cfg][cond] = []
        data[cfg][cond].append(res)
    return data


def extract_metadata(data: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
    all_runs = [r for cfg in data.values() for cond in cfg.values() for r in cond]
    rounds_set = {r["rounds"] for r in all_runs if "rounds" in r}
    data_sources = {r.get("data_source", "toy") for r in all_runs}
    he_backends = {r.get("he_backend", "mock") for r in all_runs}
    seeds_set = {r["seed"] for r in all_runs if "seed" in r}

    rounds_str = ",".join(str(x) for x in sorted(rounds_set)) if rounds_set else "unknown"
    source_str = ",".join(sorted(data_sources)) if data_sources else "unknown"
    he_str = ",".join(sorted(he_backends)) if he_backends else "unknown"
    seeds_str = ",".join(str(s) for s in sorted(seeds_set)) if seeds_set else "unknown"

    max_rounds = max(rounds_set) if rounds_set else 120
    is_reduced = bool(max_rounds < 120)

    return {
        "rounds_str": rounds_str,
        "source_str": source_str,
        "he_str": he_str,
        "seeds_str": seeds_str,
        "is_reduced": is_reduced,
        "max_rounds": max_rounds,
        "he_backend": he_str,
    }


def generate_table_1(data: Dict[str, Dict[str, List[Dict[str, Any]]]], out_dir: str = ".") -> str:
    """Generates Table I: Final Accuracy (Clean / MitM / Poison), ASR, Macro-F1, Macro-Precision."""
    meta = extract_metadata(data)
    rows = []
    header = [
        "Configuration",
        "DP",
        "Clean (%)",
        "MitM (%)",
        "Poison (%)",
        "ASR (%)",
        "Macro-F1 (%)",
        "Macro-Precision (%)",
    ]

    for cfg_key in ORDERED_CONFIGS:
        if cfg_key not in data:
            continue
        display_name, dp_enabled = CONFIG_DISPLAY[cfg_key]
        dp_str = "Yes" if dp_enabled else "No"

        clean_runs = data[cfg_key].get("clean", [])
        mitm_runs = data[cfg_key].get("mitm", [])
        poison_runs = data[cfg_key].get("poison", [])

        if not clean_runs:
            sys.exit(f"MISSING: Clean runs for configuration '{cfg_key}' are missing.")
        if not mitm_runs:
            sys.exit(f"MISSING: MitM runs for configuration '{cfg_key}' are missing.")
        if not poison_runs:
            sys.exit(f"MISSING: Poison runs for configuration '{cfg_key}' are missing.")

        clean_acc = [r["final_accuracy"] * 100 for r in clean_runs]
        mitm_acc = [r["final_accuracy"] * 100 for r in mitm_runs]
        poison_acc = [r["final_accuracy"] * 100 for r in poison_runs]
        asrs = [r["asr"] * 100 for r in poison_runs]
        f1s = [r["macro_f1"] * 100 for r in poison_runs]
        precs = [r["macro_precision"] * 100 for r in poison_runs]

        row = [
            display_name,
            dp_str,
            f"{np.mean(clean_acc):.2f} ± {np.std(clean_acc):.2f}",
            f"{np.mean(mitm_acc):.2f} ± {np.std(mitm_acc):.2f}",
            f"{np.mean(poison_acc):.2f} ± {np.std(poison_acc):.2f}",
            f"{np.mean(asrs):.2f} ± {np.std(asrs):.2f}",
            f"{np.mean(f1s):.2f} ± {np.std(f1s):.2f}",
            f"{np.mean(precs):.2f} ± {np.std(precs):.2f}",
        ]
        rows.append(row)

    title_prefix = f"REDUCED RUN: {meta['rounds_str']} rounds — " if meta["is_reduced"] else ""
    meta_line = f"*Metadata: Rounds={meta['rounds_str']} | Data={meta['source_str']} | HE={meta['he_str']} | Seeds={meta['seeds_str']}*"

    md_lines = [
        f"### {title_prefix}Table I: Final Accuracy, Attack Success Rate, Macro-F1, and Macro-Precision",
        meta_line,
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in rows:
        md_lines.append("| " + " | ".join(r) + " |")

    md_content = "\n".join(md_lines)
    with open(os.path.join(out_dir, "table_1.md"), "w", encoding="utf-8") as f:
        f.write(md_content + "\n")

    csv_lines = [f"# {title_prefix}Table I ({meta_line})", ",".join([f'"{h}"' for h in header])]
    for r in rows:
        csv_lines.append(",".join([f'"{c}"' for c in r]))
    with open(os.path.join(out_dir, "table_1.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")

    return md_content


def generate_table_2(data: Dict[str, Dict[str, List[Dict[str, Any]]]], out_dir: str = ".") -> str:
    """Generates Table II: Wall-clock time, overhead vs FedAvg-clean, peak RSS."""
    meta = extract_metadata(data)
    fedavg_clean_runs = data.get("fedavg", {}).get("clean", [])
    if not fedavg_clean_runs:
        sys.exit("MISSING: FedAvg clean runs are required to compute baseline wall-clock time.")
    base_time = float(np.mean([r["elapsed_time_s"] for r in fedavg_clean_runs]))

    header = [
        "Configuration",
        "Condition",
        "Time (s)",
        "Overhead (x vs FedAvg-clean)",
        "Peak RSS (MB)",
    ]
    rows = []

    is_mock = (meta["he_str"] == "mock")

    for cfg_key in ORDERED_CONFIGS:
        if cfg_key not in data:
            continue
        display_name, _ = CONFIG_DISPLAY[cfg_key]

        for cond in ["clean", "mitm", "poison"]:
            runs = data[cfg_key].get(cond, [])
            if not runs:
                continue
            times = [r["elapsed_time_s"] for r in runs]
            rsss = [r["peak_rss_mb"] for r in runs]

            mean_t = float(np.mean(times))
            std_t = float(np.std(times))

            if is_mock:
                overhead_str = "n/a (mock)"
            else:
                overhead = mean_t / max(base_time, 1e-6)
                overhead_str = f"{overhead:.2f}x"

            mean_rss = float(np.mean(rsss))
            std_rss = float(np.std(rsss))

            row = [
                display_name,
                cond.capitalize(),
                f"{mean_t:.2f} ± {std_t:.2f}",
                overhead_str,
                f"{mean_rss:.1f} ± {std_rss:.1f}",
            ]
            rows.append(row)

    title_prefix = f"REDUCED RUN: {meta['rounds_str']} rounds — " if meta["is_reduced"] else ""
    meta_line = f"*Metadata: Rounds={meta['rounds_str']} | Data={meta['source_str']} | HE={meta['he_str']} | Seeds={meta['seeds_str']}*"
    banner = "> [!NOTE]\n> **HE overhead not measured (MockHE). Run with `--he tenseal` for headline timing and overhead.**\n\n" if is_mock else ""

    md_lines = [
        f"### {title_prefix}Table II: Wall-Clock Execution Time, Relative Overhead, and Peak Memory (RSS)",
        meta_line,
        "",
        banner,
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in rows:
        md_lines.append("| " + " | ".join(r) + " |")

    md_content = "\n".join(md_lines)
    with open(os.path.join(out_dir, "table_2.md"), "w", encoding="utf-8") as f:
        f.write(md_content + "\n")

    csv_lines = [f"# {title_prefix}Table II ({meta_line})", ",".join([f'"{h}"' for h in header])]
    for r in rows:
        csv_lines.append(",".join([f'"{c}"' for c in r]))
    with open(os.path.join(out_dir, "table_2.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")

    return md_content


def generate_figure_2(data: Dict[str, Dict[str, List[Dict[str, Any]]]], out_dir: str = "."):
    """Generates Fig. 2: Two panels (MitM and Model Poisoning), test accuracy vs round in the exact paper style."""
    meta = extract_metadata(data)
    fedavg_clean_runs = data.get("fedavg", {}).get("clean", [])
    if not fedavg_clean_runs:
        sys.exit("MISSING: FedAvg clean runs missing for Fig. 2 benchmark curve.")

    all_hist = [[h["acc"] * 100 for h in r["history"]] for r in fedavg_clean_runs]
    benchmark_curve = np.mean(all_hist, axis=0)
    total_rounds = len(benchmark_curve)
    mark_indices = [0] + list(range(9, total_rounds, 10))
    if (total_rounds - 1) not in mark_indices:
        mark_indices.append(total_rounds - 1)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    fig, (ax_mitm, ax_poison) = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=300)
    conditions = [("mitm", "Man-in-the-Middle", ax_mitm), ("poison", "Model Poisoning", ax_poison)]
    handles_dict = {}

    for cond, title, ax in conditions:
        bench_lbl, bench_col, bench_ls, bench_m, bench_lw, bench_ms = SERIES_FIG2_INFO["bench"]
        cur_x = np.arange(1, total_rounds + 1)
        h_bench, = ax.plot(
            cur_x, benchmark_curve,
            label=bench_lbl, color=bench_col, linestyle=bench_ls, marker=bench_m,
            linewidth=bench_lw, markersize=bench_ms, markevery=mark_indices, zorder=2
        )
        handles_dict["bench"] = h_bench

        for key in ["fedavg", "gashe_only", "batchcrypt", "dp_sgd", "securedrive"]:
            runs = data.get(key, {}).get(cond, [])
            if not runs:
                continue
            histories = [[h["acc"] * 100 for h in r["history"]] for r in runs]
            mean_curve = np.mean(histories, axis=0)
            rx = np.arange(1, len(mean_curve) + 1)
            lbl, col, ls, m, lw, ms = SERIES_FIG2_INFO[key]
            h_line, = ax.plot(
                rx, mean_curve,
                label=lbl, color=col, linestyle=ls, marker=m,
                linewidth=lw, markersize=ms, markevery=mark_indices, zorder=3
            )
            handles_dict[key] = h_line

        title_prefix = f"REDUCED RUN ({meta['rounds_str']} rds): " if meta["is_reduced"] else ""
        ax.set_title(f"{title_prefix}{title}", fontsize=14, fontweight="bold", pad=10)
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

    ordered_handles = [handles_dict[k] for k in LEGEND_ORDER_FIG2 if k in handles_dict]
    ordered_labels = [SERIES_FIG2_INFO[k][0] for k in LEGEND_ORDER_FIG2 if k in handles_dict]

    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=3,
        frameon=True,
        fancybox=True,
        edgecolor="#cccccc",
        facecolor="white",
        framealpha=1.0,
        fontsize=10.5,
        handlelength=2.4,
        columnspacing=2.2,
    )

    plt.subplots_adjust(bottom=0.23, top=0.90, left=0.065, right=0.985, wspace=0.18)

    png_path = os.path.join(out_dir, "fig2.png")
    pdf_path = os.path.join(out_dir, "fig2.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Fig. 2 to {png_path} and {pdf_path}")

    # Mirror to images/ and github_repo/figures/ if available
    for extra_dir in [os.path.join(out_dir, "images"), os.path.join(out_dir, "github_repo", "figures"), "images", os.path.join("github_repo", "figures")]:
        if os.path.isdir(extra_dir):
            try:
                import shutil
                shutil.copy(png_path, os.path.join(extra_dir, "fig2.png"))
                shutil.copy(pdf_path, os.path.join(extra_dir, "fig2.pdf"))
            except Exception:
                pass


def make_full_report(results_dir: str = "results", out_dir: str = "."):
    data = load_all_results(results_dir)
    print("Generating Table I...")
    t1 = generate_table_1(data, out_dir)
    print(t1)
    print("\n" + "=" * 80 + "\n")

    print("Generating Table II...")
    t2 = generate_table_2(data, out_dir)
    print(t2)
    print("\n" + "=" * 80 + "\n")

    print("Generating Fig. 2...")
    generate_figure_2(data, out_dir)


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    make_full_report(results_dir, out_dir)
