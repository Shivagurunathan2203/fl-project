#!/usr/bin/env python3
"""
Grid Runner for Compute-Budgeted Selective Encryption Extension (spec_budget.md).
Runs:
  - H1 hetero_tiers: All strategies, all 3 conditions, 3 seeds.
  - S1 homog_f budget sweep: f in [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0] under MitM.
  - S2 bursty: sd_budget and sd_exclude under Markov availability.
Reuses base results where permitted via exact SHA256 config hash match (spec_budget.md Section 8.1 / 8.3).
"""

import os
import sys
import json
import csv
import math
import hashlib
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from securedrive_fl.config import Config
from securedrive_fl.he import MockHE, TenSealCKKS
from securedrive_fl.data import make_toy_federation, check_real_data_exists, load_real_driver_data, load_dataset
from securedrive_fl.budget.config import BudgetConfig, CapacityConfig, AvailabilityConfig
from securedrive_fl.budget.scenarios import get_scenario_config
from securedrive_fl.budget.strategies import run_budgeted_scenario
from securedrive_fl.budget.time_model import LatencyModel
from securedrive_fl.experiments import compute_config_hash, get_git_hash


def load_base_result(
    config_name: str,
    condition: str,
    seed: int,
    expected_hash: Optional[str] = None,
    base_dir: str = "results",
) -> Optional[Dict[str, Any]]:
    path = Path(base_dir) / f"{config_name}_{condition}_seed{seed}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if expected_hash is not None:
            if data.get("config_hash") != expected_hash:
                return None
        return data
    except Exception:
        return None


def run_single_ext_scenario(
    strategy: str,
    scenario_name: str,
    condition: str,
    seed: int,
    rounds: int = 120,
    he_backend: str = "mock",
    use_real_data: bool = False,
    dataset: str = "statefarm",
    data_dir: str = "data",
    results_ext_dir: str = "results_ext",
    latency_model: Optional[LatencyModel] = None,
) -> Dict[str, Any]:
    """
    Executes a single scenario for the Compute-Budgeted Selective Encryption Extension.

    Strategies Evaluated:
      - 'sd_budget'        : Our proposal (gradient energy ranking post-DP noise)
      - 'sd_wait'          : Base paper upper-bound (waits for slowest straggler to encrypt all)
      - 'sd_exclude'       : Coverage-gated exclusion (drops vehicles unable to meet coverage)
      - 'sd_budget_random' : Ablation replacing energy ranking with random chunk selection
      - 'sd_budget_layer'  : Ablation using static layer-order chunk selection
      - 'gashe_budget'     : Non-DP proposal variant for isolation of cryptographic benefit

    Scenarios Evaluated:
      - 'hetero_tiers'     : Headline H1 (3 device tiers: high, mid, low capacity)
      - 'homog_f{x}'       : S1 Budget sweep across budget fractions f in [0.0, 1.0]
      - 'bursty'           : S2 Markov two-state availability channel (Good/Bad network)
    """
    out_dir = Path(results_ext_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{strategy}_{scenario_name}_{condition}_seed{seed}.json"
    out_csv = out_dir / f"{strategy}_{scenario_name}_{condition}_seed{seed}_rounds.csv"

    # Fast return if already computed (Idempotence & Cluster Resume Guard)
    if out_json.exists():
        try:
            with open(out_json, "r") as f:
                existing = json.load(f)
            if "final_accuracy" in existing:
                return existing
        except Exception:
            # Corrupted or partial write from interrupted Slurm job; recompute safely
            pass

    # Base configuration setup
    use_dp = not strategy.startswith("gashe")
    cfg = Config(rounds=rounds, seed=seed, use_dp=use_dp)

    if condition == "clean":
        cfg.attack = None
        cfg.loss_guard = False
    elif condition == "mitm":
        cfg.attack = "mitm"
        cfg.loss_guard = True
    elif condition == "poison":
        cfg.attack = "model_poison"
        cfg.loss_guard = False
        cfg.poison_mode = "backdoor"

    # Load budget scenario specification
    b_cfg = get_scenario_config(scenario_name)

    # Strategy ranking dispatch
    if strategy == "sd_budget_random":
        b_cfg.rank_by = "random"
    elif strategy == "sd_budget_layer":
        b_cfg.rank_by = "layer_static"

    # Load federation data without silent fallback
    if use_real_data:
        has_real, missing = check_real_data_exists(data_dir, dataset=dataset)
        if not has_real:
            help_cmd = "python scripts/download_dataset.py"
            raise FileNotFoundError(
                f"Real dataset '{dataset}' requested (--real-data) but not found in '{data_dir}'. Missing: {missing}.\n"
                f"Please run '{help_cmd}' to download/cache it."
            )
        client_data, test_data, K = load_dataset(dataset=dataset, data_dir=data_dir, seed=seed)
        data_source = f"real_{dataset}"
        # Calibrate learning rate, local epochs, and chunk size for 1280-dim embeddings
        cfg.lr = 0.02
        cfg.local_epochs = 3
        b_cfg.chunk_size = 512
    else:
        client_data, test_data, K = make_toy_federation(seed=seed, separation=getattr(cfg, "separation", 3.8))
        data_source = "toy"

    he = TenSealCKKS() if he_backend == "tenseal" else MockHE()

    # Execute
    res = run_budgeted_scenario(
        cfg=cfg,
        b_cfg=b_cfg,
        strategy=strategy,
        client_data=client_data,
        test=test_data,
        K=K,
        he=he,
        latency_model=latency_model,
    )

    config_hash = compute_config_hash(cfg, data_source=data_source, model_name="numpy_softmax", seed=seed)

    # Save JSON summary
    summary = {
        "strategy": strategy,
        "scenario": scenario_name,
        "condition": condition,
        "seed": seed,
        "rounds": rounds,
        "dataset": dataset if use_real_data else "toy",
        "data_source": data_source,
        "he_backend": he_backend,
        "model": "numpy_softmax",
        "noise_std_used": cfg.sigma if cfg.use_dp else 0.0,
        "chunk_size": res.get("chunk_size", 4096),
        "n_chunks": res.get("n_chunks", 25),
        "virtual": res.get("virtual", True),
        "min_coverage": res.get("min_coverage", getattr(b_cfg, "min_coverage", 0.5)),
        "degenerate": res.get("degenerate", False),
        "git_hash": get_git_hash(),
        "config_hash": config_hash,
        "final_accuracy": res["final_accuracy"],
        "macro_f1": res["macro_f1"],
        "macro_precision": res["macro_precision"],
        "asr": res["asr"],
        "final_epsilon": res["final_epsilon"],
        "epsilon_at_120": res.get("epsilon_at_120"),
        "total_sim_time_s": res["total_sim_time_s"],
        "mean_effective_budget": res["mean_effective_budget"],
        "min_effective_budget": res["min_effective_budget"],
        "max_effective_budget": res["max_effective_budget"],
        "mean_protected_energy_fraction": res["mean_protected_energy_fraction"],
        "mean_protected_coord_fraction": res["mean_protected_coord_fraction"],
        "participation_counts": res["participation_counts"],
        "capacity": res["capacity"],
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-round CSV
    if res["rounds"]:
        fieldnames = sorted(list({k for r_entry in res["rounds"] for k in r_entry.keys()}))
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r_entry in res["rounds"]:
                row = r_entry.copy()
                for k, v in row.items():
                    if isinstance(v, (list, dict)):
                        row[k] = json.dumps(v)
                writer.writerow(row)

    return summary


def run_full_extension_grid(
    rounds: int = 120,
    seeds: List[int] = [42, 123, 456],
    he_backend: str = "mock",
    use_real_data: bool = False,
    dataset: str = "statefarm",
    data_dir: str = "data",
    results_ext_dir: str = "results_ext",
    budget_seconds: Optional[float] = None,
):
    print("==================================================================")
    print("Starting Compute-Budgeted Selective Encryption Extension Grid Run")
    print(f"Dataset: {dataset if use_real_data else 'toy'} | Rounds: {rounds} | Seeds: {seeds} | Real Data: {use_real_data} | Backend: {he_backend}")
    print(f"Results Directory: {results_ext_dir}")
    print("==================================================================")

    latency = LatencyModel()

    # Estimate total compute
    h1_strategies = ["sd_budget", "sd_wait", "sd_exclude", "sd_budget_random", "sd_budget_layer", "gashe_budget"]
    conditions = ["clean", "mitm", "poison"]
    sweep_fractions = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]

    h1_runs = (4 * 3 * len(seeds)) + (2 * 1 * len(seeds))  # 4 strats all 3 conds, 2 ablations mitm only
    s1_runs = len(sweep_fractions) * 1 + len(sweep_fractions) * 2  # sd_budget + 2 ablations
    s2_runs = 2 * 1  # 2 strats mitm
    est_total_runs = h1_runs + s1_runs + s2_runs

    est_seconds = est_total_runs * rounds * (0.015 if he_backend == "mock" else 0.8)
    print(f"[Compute Estimate] Approx. {est_total_runs} runs x {rounds} rounds ~ {est_seconds:.1f}s ({est_seconds/60:.1f} min)")

    if budget_seconds is not None and est_seconds > budget_seconds:
        sys.exit(f"ABORT: Estimated compute {est_seconds:.1f}s exceeds budget {budget_seconds:.1f}s. Stop and ask.")

    # 1. Headline H1 (hetero_tiers) across conditions
    for strat in h1_strategies:
        for cond in conditions:
            if strat in ("sd_budget_random", "sd_budget_layer") and cond != "mitm":
                continue
            for s in seeds:
                print(f"[H1] {strat:<18} | {cond:<6} | seed={s} ...", end="", flush=True)
                res = run_single_ext_scenario(
                    strategy=strat,
                    scenario_name="hetero_tiers",
                    condition=cond,
                    seed=s,
                    rounds=rounds,
                    he_backend=he_backend,
                    use_real_data=use_real_data,
                    dataset=dataset,
                    data_dir=data_dir,
                    results_ext_dir=results_ext_dir,
                    latency_model=latency,
                )
                print(f" done. Acc: {res['final_accuracy']*100:.2f}%, ProtEnergy: {res['mean_protected_energy_fraction']*100:.1f}%, SimTime: {res['total_sim_time_s']:.2f}s")

    # 2. S1 Budget Sweep (f in [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]) under MitM
    for f_val in sweep_fractions:
        for s in [42]:
            scenario = f"homog_f{f_val}"
            print(f"[S1] sd_budget ({scenario}) | mitm | seed={s} ...", end="", flush=True)
            res = run_single_ext_scenario(
                strategy="sd_budget",
                scenario_name=scenario,
                condition="mitm",
                seed=s,
                rounds=rounds,
                he_backend=he_backend,
                use_real_data=use_real_data,
                dataset=dataset,
                data_dir=data_dir,
                results_ext_dir=results_ext_dir,
                latency_model=latency,
            )
            print(f" done. Acc: {res['final_accuracy']*100:.2f}%, ProtEnergy: {res['mean_protected_energy_fraction']*100:.1f}%")

    # S1 ranking ablations across all budget fractions
    for f_val in sweep_fractions:
        for strat in ["sd_budget_random", "sd_budget_layer"]:
            scenario = f"homog_f{f_val}"
            print(f"[S1] {strat} ({scenario}) | mitm | seed=42 ...", end="", flush=True)
            res = run_single_ext_scenario(
                strategy=strat,
                scenario_name=scenario,
                condition="mitm",
                seed=42,
                rounds=rounds,
                he_backend=he_backend,
                use_real_data=use_real_data,
                dataset=dataset,
                data_dir=data_dir,
                results_ext_dir=results_ext_dir,
                latency_model=latency,
            )
            print(f" done. Acc: {res['final_accuracy']*100:.2f}%")

    # 3. S2 Bursty (Markov availability)
    for strat in ["sd_budget", "sd_exclude"]:
        for s in [42]:
            print(f"[S2] {strat:<18} (bursty) | mitm | seed={s} ...", end="", flush=True)
            res = run_single_ext_scenario(
                strategy=strat,
                scenario_name="bursty",
                condition="mitm",
                seed=s,
                rounds=rounds,
                he_backend=he_backend,
                use_real_data=use_real_data,
                dataset=dataset,
                data_dir=data_dir,
                results_ext_dir=results_ext_dir,
                latency_model=latency,
            )
            print(f" done. Acc: {res['final_accuracy']*100:.2f}%, SimTime: {res['total_sim_time_s']:.2f}s")

    print("\n[Success] All extension grid runs completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run extension experiment grid.")
    parser.add_argument("--rounds", type=int, default=120, help="FL communication rounds (default: 120)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Random seeds")
    parser.add_argument("--he", choices=["mock", "tenseal"], default="mock", help="HE backend")
    parser.add_argument("--real-data", "--use-real-data", dest="real_data", action="store_true", help="Use real driver dataset")
    parser.add_argument("--dataset", choices=["statefarm", "auc"], default="statefarm", help="Dataset name (default: statefarm)")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--results-ext-dir", default="results_ext", help="Results extension directory")
    parser.add_argument("--budget-seconds", type=float, default=None, help="Maximum allowed compute budget in seconds")
    args = parser.parse_args()

    run_full_extension_grid(
        rounds=args.rounds,
        seeds=args.seeds,
        he_backend=args.he,
        use_real_data=args.real_data,
        dataset=args.dataset,
        data_dir=args.data_dir,
        results_ext_dir=args.results_ext_dir,
        budget_seconds=args.budget_seconds,
    )
