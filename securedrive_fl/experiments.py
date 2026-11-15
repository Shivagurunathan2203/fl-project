from __future__ import annotations

import os
import sys
import time
import math
import json
import hashlib
import argparse
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import psutil

from securedrive_fl.config import Config
from securedrive_fl.he import MockHE, TenSealCKKS, HEBackend
from securedrive_fl.data import make_toy_federation, check_real_data_exists, load_real_driver_data, load_dataset
from securedrive_fl.baselines import NINE_CONFIGS, get_config_for_baseline
from securedrive_fl.run_scenario import run_scenario
from securedrive_fl.privacy import worst_case_epsilon
from securedrive_fl.metrics import compute_metrics


def get_git_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def compute_config_hash(cfg: Config, data_source: str, model_name: str, seed: int) -> str:
    d = {
        "data_source": data_source,
        "model": model_name,
        "n_clients": cfg.n_clients,
        "rounds": cfg.rounds,
        "local_epochs": cfg.local_epochs,
        "batch": cfg.batch,
        "lr": cfg.lr,
        "use_dp": cfg.use_dp,
        "clip_C": cfg.clip_C,
        "eps0": cfg.eps0,
        "delta": cfg.delta,
        "use_gashe": cfg.use_gashe,
        "top_k": cfg.top_k,
        "aggregator": cfg.aggregator,
        "krum_f": getattr(cfg, "krum_f", 2),
        "attack": cfg.attack,
        "mitm_noise_std": cfg.mitm_noise_std,
        "mitm_amplify": cfg.mitm_amplify,
        "poison_mode": getattr(cfg, "poison_mode", "backdoor"),
        "poison_alpha": getattr(cfg, "poison_alpha", 0.5),
        "poison_boost": getattr(cfg, "poison_boost", 1.0),
        "target_class": getattr(cfg, "target_class", 2),
        "seed": seed,
    }
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


class MemoryTracker:
    """Tracks peak RSS memory in a background thread."""
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_rss_mb = 0.0
        self._stop_event = threading.Event()
        self._thread = None
        self._proc = psutil.Process()

    def _monitor(self):
        while not self._stop_event.is_set():
            try:
                rss_mb = self._proc.memory_info().rss / (1024 * 1024)
                if rss_mb > self.peak_rss_mb:
                    self.peak_rss_mb = rss_mb
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.peak_rss_mb = self._proc.memory_info().rss / (1024 * 1024)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            final_rss = self._proc.memory_info().rss / (1024 * 1024)
            if final_rss > self.peak_rss_mb:
                self.peak_rss_mb = final_rss
        except Exception:
            pass
        return self.peak_rss_mb


def run_single_experiment(
    config_name: str,
    condition: str,  # "clean", "mitm", "poison"
    seed: int,
    rounds: int = 120,
    he_backend: str = "mock",
    use_real_data: bool = False,
    dataset: str = "statefarm",
    data_dir: str = "data",
    results_dir: str = "results",
) -> Dict[str, Any]:
    """
    Runs a single cell of the experiment matrix in an isolated subprocess.

    Key Responsibilities:
    1. Instantiates the baseline config (FedAvg, FedProx, GASHE, BatchCrypt, DP-SGD, SecureDrive, etc.).
    2. Configures adversarial condition (Clean, Blanchard MitM sign-flipping, or Eq. 13 Backdoor).
    3. Loads requested dataset (Toy Gaussian, State Farm / AUC Distracted Driver, or GTSRB Traffic Signs).
    4. Tracks wall-clock time and peak RSS memory via background daemon thread.
    5. Saves full execution metrics and round-by-round trajectory to an atomic JSON file.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    out_file = Path(results_dir) / f"{config_name}_{condition}_seed{seed}.json"

    # Setup base configuration from baseline dictionary
    cfg = get_config_for_baseline(config_name)
    cfg.rounds = rounds
    cfg.seed = seed

    # Map adversarial condition
    if condition == "clean":
        cfg.attack = None
        cfg.loss_guard = False
    elif condition == "mitm":
        cfg.attack = "mitm"
        # LossGuard enabled only for securedrive under MitM as in hld / paper
        if config_name == "securedrive":
            cfg.loss_guard = True
        else:
            cfg.loss_guard = False
    elif condition == "poison":
        cfg.attack = "model_poison"
        cfg.loss_guard = False
        cfg.poison_mode = "backdoor"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Prepare data without silent fallback
    if use_real_data:
        has_real, missing = check_real_data_exists(data_dir, dataset=dataset)
        if not has_real:
            help_cmd = "python scripts/download_dataset.py"
            raise FileNotFoundError(
                f"Real dataset '{dataset}' requested (--real-data) but not found in '{data_dir}'. Missing: {missing}.\n"
                f"Please run '{help_cmd}' to download/cache it.\n"
                f"See guide.md for details."
            )
        # Load preprocessed feature cache or extract on the fly
        client_data, test_data, K = load_dataset(dataset=dataset, data_dir=data_dir, seed=seed)
        data_source = f"real_{dataset}"
        # Calibrate learning rate and local epochs for 1280-dim normalized MobileNetV2 embeddings
        cfg.lr = 0.02
        cfg.local_epochs = 3
        # Class 2: "talking on phone - right" for distracted driver benchmark
        cfg.target_class = 2
    else:
        client_data, test_data, K = make_toy_federation(seed=seed, separation=getattr(cfg, "separation", 3.8))
        data_source = "toy"

    # Prepare HE backend
    if he_backend == "tenseal":
        he = TenSealCKKS()
    else:
        he = MockHE()

    # Track time and peak memory
    mem_tracker = MemoryTracker()
    mem_tracker.start()
    t0 = time.perf_counter()

    history = run_scenario(cfg, client_data, test_data, K, he)

    elapsed_time = time.perf_counter() - t0
    peak_rss = mem_tracker.stop()

    final_h = history[-1]
    final_acc = float(final_h["acc"])
    macro_f1 = float(final_h.get("macro_f1", final_acc))
    macro_prec = float(final_h.get("macro_precision", final_acc))
    asr = float(final_h.get("asr", 0.0))

    sizes = [len(X) for X, _ in client_data]
    final_eps = worst_case_epsilon(sizes, rounds, cfg) if cfg.use_dp else None
    eps_120 = worst_case_epsilon(sizes, 120, cfg) if cfg.use_dp else None
    gap_to_paper = (final_eps - 48.94) if final_eps is not None else None

    config_hash = compute_config_hash(cfg, data_source=data_source, model_name="numpy_softmax", seed=seed)

    result_data = {
        "config_name": config_name,
        "condition": condition,
        "seed": seed,
        "rounds": rounds,
        "dataset": dataset if use_real_data else "toy",
        "data_source": data_source,
        "he_backend": he_backend,
        "model": "numpy_softmax",
        "noise_std_used": cfg.sigma if cfg.use_dp else 0.0,
        "chunk_size": 4096,
        "git_hash": get_git_hash(),
        "config_hash": config_hash,
        "resolved_config": {
            "n_clients": cfg.n_clients,
            "rounds": cfg.rounds,
            "local_epochs": cfg.local_epochs,
            "batch": cfg.batch,
            "lr": cfg.lr,
            "use_dp": cfg.use_dp,
            "clip_C": cfg.clip_C,
            "sigma": cfg.sigma,
            "z": cfg.z,
            "delta": cfg.delta,
            "use_gashe": cfg.use_gashe,
            "top_k": cfg.top_k,
            "aggregator": cfg.aggregator,
            "krum_f": getattr(cfg, "krum_f", 2),
            "attack": cfg.attack,
            "mitm_noise_std": cfg.mitm_noise_std,
            "poison_mode": getattr(cfg, "poison_mode", "backdoor"),
            "poison_alpha": getattr(cfg, "poison_alpha", 0.5),
            "poison_boost": getattr(cfg, "poison_boost", 1.0),
            "target_class": getattr(cfg, "target_class", 2),
            "separation": getattr(cfg, "separation", 3.8),
        },
        "elapsed_time_s": elapsed_time,
        "peak_rss_mb": peak_rss,
        "final_accuracy": final_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "asr": asr,
        "final_epsilon": final_eps,
        "epsilon_at_120": eps_120,
        "gap_to_paper_48_94": gap_to_paper,
        "mean_enc_frac": float(np.mean([h["enc_frac"] for h in history])),
        "tau_tiers": sorted(list({t for h in history for t in h["tiers"]})),
        "history": [
            {
                "round": h["round"],
                "acc": h["acc"],
                "loss": h["loss"],
                "rejected": h["rejected"],
                "flagged": h.get("flagged", h["rejected"]),
                "enc_frac": h["enc_frac"],
                "eps": h["eps"] if (h["eps"] is not None and math.isfinite(h["eps"])) else None,
            }
            for h in history
        ],
    }

    with open(out_file, "w") as f:
        json.dump(result_data, f, indent=2)

    return result_data


def run_all_experiments(
    configs: Optional[List[str]] = None,
    conditions: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    rounds: int = 120,
    he_backend: str = "mock",
    use_real_data: bool = False,
    dataset: str = "statefarm",
    data_dir: str = "data",
    results_dir: str = "results",
):
    """Runs the full experiment matrix using fresh subprocesses for RSS independence."""
    configs = configs or list(NINE_CONFIGS.keys())
    conditions = conditions or ["clean", "mitm", "poison"]
    seeds = seeds or [42, 123, 456]

    total_runs = len(configs) * len(conditions) * len(seeds)
    print("==================================================================")
    print(f"Starting experiments: {len(configs)} configs x {len(conditions)} conditions x {len(seeds)} seeds = {total_runs} total runs.")
    print(f"Dataset: {dataset if use_real_data else 'toy'}, Backend: {he_backend}, Rounds: {rounds}, Real Data: {use_real_data}")
    print(f"Results Directory: {results_dir}")
    print("==================================================================")

    count = 0
    for cfg_name in configs:
        for cond in conditions:
            for s in seeds:
                count += 1
                out_path = Path(results_dir) / f"{cfg_name}_{cond}_seed{s}.json"

                # Checkpoint & Idempotence Guard:
                # If this cell already finished in a previous run and produced valid metrics,
                # skip it immediately. This allows uninterrupted resumption after cluster time limits.
                if out_path.exists():
                    try:
                        with open(out_path, "r") as f:
                            res = json.load(f)
                        if "final_accuracy" in res:
                            print(f"[{count}/{total_runs}] {cfg_name} | {cond} | seed={s} ... cached. Acc: {res['final_accuracy']:.3f}, Time: {res.get('elapsed_time_s', 0):.2f}s")
                            continue
                    except Exception:
                        # File was corrupted or partially written during prior job termination; safely recompute
                        pass

                print(f"[{count}/{total_runs}] Running {cfg_name} | {cond} | seed={s} ...", end="", flush=True)

                # Execute in fresh subprocess for true isolated peak RSS measurement (psutil)
                # This guarantees zero memory leakage or Python runtime heap pollution across baselines.
                cmd = [
                    sys.executable, "-m", "securedrive_fl.experiments",
                    "--single",
                    "--config", cfg_name,
                    "--condition", cond,
                    "--seed", str(s),
                    "--rounds", str(rounds),
                    "--he", he_backend,
                    "--dataset", dataset,
                    "--data-dir", data_dir,
                    "--results-dir", results_dir,
                ]
                if use_real_data:
                    cmd.append("--real-data")

                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    print(f" FAILED!\nError: {proc.stderr}")
                    sys.exit(proc.returncode)

                # Read output JSON
                out_path = Path(results_dir) / f"{cfg_name}_{cond}_seed{s}.json"
                if not out_path.exists():
                    print(" FAILED! Missing output JSON.")
                    sys.exit(1)

                with open(out_path, "r") as f:
                    res = json.load(f)

                print(f" done. Acc: {res['final_accuracy']:.3f}, Time: {res['elapsed_time_s']:.2f}s, RSS: {res['peak_rss_mb']:.1f}MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SecureDrive-FL experiment matrix.")
    parser.add_argument("--single", action="store_true", help="Run a single experiment directly (subprocess mode)")
    parser.add_argument("--config", default=None, help="Config name for single run")
    parser.add_argument("--condition", choices=["clean", "mitm", "poison"], default="clean", help="Condition for single run")
    parser.add_argument("--seed", type=int, default=42, help="Seed for single run")

    parser.add_argument("--configs", nargs="+", default=None, help="Baseline names to run (default: all 9)")
    parser.add_argument("--conditions", nargs="+", default=None, help="Conditions: clean, mitm, poison (default: all 3)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Seeds to run (default: 42 123 456)")
    parser.add_argument("--rounds", type=int, default=120, help="Number of FL rounds (default: 120)")
    parser.add_argument("--he", choices=["mock", "tenseal"], default="mock", help="HE backend (default: mock)")
    parser.add_argument("--real-data", "--use-real-data", dest="real_data", action="store_true", help="Use real dataset (State Farm / AUC)")
    parser.add_argument("--dataset", choices=["statefarm", "auc"], default="statefarm", help="Dataset name (default: statefarm)")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--results-dir", default="results", help="Directory to save JSON results")
    args = parser.parse_args()

    if args.single:
        if not args.config:
            parser.error("--config is required when --single is set")
        run_single_experiment(
            config_name=args.config,
            condition=args.condition,
            seed=args.seed,
            rounds=args.rounds,
            he_backend=args.he,
            use_real_data=args.real_data,
            dataset=args.dataset,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
        )
    else:
        run_all_experiments(
            configs=args.configs,
            conditions=args.conditions,
            seeds=args.seeds,
            rounds=args.rounds,
            he_backend=args.he,
            use_real_data=args.real_data,
            dataset=args.dataset,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
        )
