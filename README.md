# SecureDrive-FL: Privacy-Preserving & Byzantine-Robust Federated Learning for Connected Autonomous Vehicles

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![TenSEAL](https://img.shields.io/badge/TenSEAL-CKKS-green.svg)](https://github.com/OpenMined/TenSEAL)
[![Tests](https://img.shields.io/badge/Tests-33%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An open-source implementation and extension of **SecureDrive-FL**, a federated learning framework designed for connected autonomous vehicles (CAVs) facing non-IID edge data, channel eavesdropping, Byzantine gradient poisoning, and Man-in-the-Middle (MitM) parameter tampering.

This repository provides:
1. **Full Paper Reproduction**: End-to-end implementation of DP-SGD with Gradient-Aware Selective Homomorphic Encryption (GASHE), TenSEAL CKKS encryption, dynamic thresholding, and server-side 3-$\sigma$ Loss Guard.
2. **Compute-Budgeted Extension**: Dynamic chunked selective encryption addressing real-world vehicular hardware heterogeneity (compute-constrained on-board units) to maximize Byzantine protection within per-round latency deadlines.


---

## 1. System Architecture

SecureDrive-FL operates a dual-stream cryptographic decomposition on local model updates:

```
                      +-----------------------------+
                      |   Client Local Training     |
                      |  (MobileNetV2-Tiny, DP-SGD) |
                      +--------------+--------------+
                                     |
                         Local Update \Delta w_i
                                     |
                       +-------------v-------------+
                       |   GASHE Threshold Mask    |
                       |       (|w_j| >= \tau)      |
                       +------+-------------+------+
                              |             |
        Top Coordinates (Encrypted)         Plaintext Residue (Differentially Private)
                              |             |
             c_i = Enc(M \odot \Delta w)    p_i = (1 - M) \odot \Delta w
                              |             |
                              +------+------+
                                     |
                            Over-the-Air Payload
                                     v
                       +---------------------------+
                       |      Untrusted Server     |
                       |   (No Secret Key Access)  |
                       +-------------+-------------+
                                     |
                 Homomorphic Addition: C_agg = \sum c_i
                 Plaintext Vector Sum: P_agg = \sum p_i
                 Loss Guard Evaluation: L_{t+1} <= L_t + 3\sigma
                                     |
                                     v
                       +---------------------------+
                       |   Client-Side Decryption  |
                       | W <- W + (Dec(C) + P) / N |
                       +---------------------------+
```

### Key Technical Highlights
- **Cryptographic Asymmetry**: The server only holds the TenSEAL CKKS public encryption context and cannot decrypt individual client coordinates. Decryption occurs strictly on client vehicles holding secret keys.
- **Strict Mask Partitioning**: Encrypted coordinates and plaintext coordinates are disjoint (`np.where(M, dw, 0)` and `np.where(M, 0, dw)`), preventing gradient cross-leakage.
- **Compute-Budgeted Selective Encryption (`sd_budget`)**: When vehicles have heterogeneous compute capacities (e.g. 10%, 25%, 50%, 100% of nominal encryption budget), clients rank gradient chunks by L2 magnitude and encrypt only the most influential chunks within their deadline, sending the remaining coordinates with DP noise.

---

## 2. Experimental Results Summary

### Base Paper Reproduction (120 Rounds, 6 Clients, State Farm MobileNetV2 Features)

| Method | Clean Acc (%) | MitM Acc (%) | Poison Acc (%) | ASR (%) | Marginal HE Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FedAvg** | 95.39% | 10.51% (Collapse) | 94.88% | 98.42% | Baseline (0.00%) |
| **Trimmed Mean** | 94.02% | 10.22% (Collapse) | 93.92% | 95.84% | 0.00% |
| **Multi-Krum** | 91.85% | 10.15% (Collapse) | 91.54% | 94.61% | 0.00% |
| **FedProx** | 94.86% | 10.48% (Collapse) | 94.40% | 98.20% | 0.00% |
| **BatchCrypt** | 95.12% | 95.10% | 94.65% | 98.15% | +164.20% |
| **DP-SGD** | 78.42% | 10.18% (Collapse) | 78.89% | 3.12% | +0.42% |
| **GASHE Top-60%** | 95.22% | 77.10% | 94.50% | 98.30% | +4.61% |
| **SecureDrive-FL (Ours)**| **78.25%** | **77.82%** | **79.02%** | **2.88%** | **+4.66%** |

*Key Takeaway*: FedAvg, Trimmed Mean, Krum, and FedProx completely collapse to random guessing (~10%) under MitM tampering. DP-SGD eliminates backdoor poisoning (ASR drops to 3.12%) but remains vulnerable to channel tampering. SecureDrive-FL concurrently defends against both MitM (77.82%) and backdoor poisoning (2.88% ASR) with only 4.66% marginal overhead over plain DP-SGD.

---

### Compute-Budgeted Heterogeneity (Scenario H1: Heavy Stragglers, Mean Budget ~28%)

| Strategy | MitM Acc (%) | Poison Acc (%) | Wall-Clock Time (s) | Straggler Wait (s) | Participation Fairness |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Drop Weak (`sd_exclude`)** | 40.04% | 40.12% | 23.48 s | 0.00 s | 0.00 (Exclusion) |
| **Wait for All (`sd_wait`)** | 85.04% | 85.64% | 333.67 s | 310.18 s | 1.00 |
| **Budgeted Encryption (`sd_budget`)** | **84.04%** | **84.66%** | **23.49 s** | **0.00 s** | **1.00 (Perfect)** |

*Key Takeaway*: Dropping weak clients (`sd_exclude`) causes severe catastrophic forgetting and model collapse (40.04% accuracy, 0.00 fairness). Waiting for weak vehicles (`sd_wait`) incurs a 14.2x latency penalty. Our budgeted selective encryption (`sd_budget`) delivers near-optimal accuracy (84.04%) in real time (23.49s) with zero straggler delay and 100% vehicle inclusion.

---

## 3. Repository Structure

```
sdfl_repo/
├── securedrive_fl/              # Core framework package
│   ├── __init__.py              # Public exports
│   ├── config.py                # Hyperparameter dataclasses
│   ├── he.py                    # MockHE & TenSealCKKS backend wrappers
│   ├── gashe.py                 # Mask generation and threshold tiers
│   ├── aggregate.py             # Server homomorphic and plaintext aggregation
│   ├── attacks.py               # MitM channel noise and backdoor injection
│   ├── guard.py                 # 3-sigma LossGuard filter
│   ├── privacy.py               # RDP privacy accountant
│   ├── train.py                 # Local client training and evaluation
│   ├── model.py                 # MobileNetV2-Tiny architecture
│   ├── data.py                  # Dataset partitioner and toy federation
│   ├── baselines.py             # 9 baseline configurations
│   ├── metrics.py               # Accuracy, Macro-F1, Precision, ASR
│   ├── run_scenario.py          # Unified round execution loop
│   ├── experiments.py           # Experiment matrix runner
│   ├── make_report.py           # Base paper figure & table generator
│   └── budget/                  # Compute-budgeted extension package
│       ├── __init__.py
│       ├── config.py            # Budget & latency configurations
│       ├── model.py             # Client capacity & availability models
│       ├── codec.py             # Chunked selective encryption codec
│       ├── aggregate.py         # Budgeted payload aggregation
│       ├── strategies.py        # sd_budget, sd_wait, sd_exclude runners
│       ├── inversion.py         # DLG gradient leakage audit
│       ├── time_model.py        # Straggler and round latency model
│       └── scenarios.py         # Heterogeneity scenarios (H1, S1, S2)
├── scripts/                     # Cluster execution and plotting scripts
│   ├── calibrate_he_cost.py     # Measure empirical CKKS encryption latency
│   ├── download_statefarm_dataset.py
│   ├── download_auc_dataset.py
│   ├── run_grid_ext.py          # Grid runner for extension sweeps
│   ├── run_audit.py             # Privacy leakage DLG audit runner
│   ├── make_figures_ext.py      # Generate Figures 2-6 (PNG and PDF)
│   ├── make_figure_computation_cost.py # Compute cost vs rounds plots
│   ├── make_algorithm_diagram.py # Generate protocol ladder diagram (TikZ)
│   ├── make_tables_ext.py       # Generate Markdown & CSV tables
│   ├── make_claims_report.py    # Formal claim verification checker
│   ├── plot_mitm_analysis.py    # Detailed MitM degradation plots
│   ├── submit_gpu.sh            # Slurm cluster submission script (State Farm)
│   └── submit_extension.sh      # Slurm extension script
├── tests/                       # Formal verification test suite
│   ├── test_securedrive.py      # Core protocol unit & integration tests
│   └── test_budget_invariants.py # Extension invariants and safety tests
├── docs/                        # Complete technical documentation
│   ├── implement.md             # Complete viva defense implementation guide
│   ├── claims_report.md         # Formal verification report (8/8 claims hold)
│   ├── RESULTS.md               # Base experimental results discussion
│   ├── REPORT_EXT.md            # Extension experimental results discussion
│   ├── DEVIATIONS_EXT.md        # Technical calibration deviations
│   ├── report.tex               # LaTeX source of research report
│   └── report.pdf               # Compiled 7-page IEEE/ACM research paper
├── figures/                     # All generated plots and visual comparisons
├── tables/                      # Raw result CSV and Markdown tables
├── data/                        # Dataset placeholder and instructions
├── demo.py                      # 15-second quickstart demo
├── conftest.py                  # Pytest environment bootstrap
├── requirements.txt             # Pip dependencies
├── environment.yml              # Conda environment definition
├── LICENSE                      # MIT License
└── README.md                    # Project documentation
```

---

## 4. Quickstart Guide

### Environment Installation

Using Conda:
```bash
conda env create -f environment.yml
conda activate sdfl_env
```

Or using Pip:
```bash
pip install -r requirements.txt
```

### Run Fast Demonstration (15 seconds)
Execute the toy federation demo across all key defense scenarios:
```bash
python demo.py
```

### Run Formal Test Suite
Verify all 33 mathematical and cryptographic invariants:
```bash
pytest -v
```

---

## 5. Reproducing Paper & Extension Experiments

### 1. Calibrate Hardware Latency
Benchmark TenSEAL CKKS encryption latency on your host machine:
```bash
python scripts/calibrate_he_cost.py
```

### 2. Execute Extension Grid Sweeps
Run heterogeneous scenario sweeps across seeds 42, 123, 456:
```bash
python scripts/run_grid_ext.py --scenario H1 --seeds 42 123 456
```

### 3. Run Privacy Leakage Audit
Evaluate Deep Leakage from Gradients (DLG) reconstruction across encryption fractions:
```bash
python scripts/run_audit.py
```

### 4. Regenerate Figures and Tables
```bash
python scripts/make_figures_ext.py
python scripts/make_tables_ext.py
python scripts/make_claims_report.py
```

All output plots are saved to `figures/` and tables to `tables/`.

---

## 5. Detailed Documentation

For an in-depth explanation of every mathematical formula, design choice, and viva defense question, consult:
- **[docs/report.pdf](docs/report.pdf)**: Complete research report formatted for ACM/IEEE publication.
- **[docs/claims_report.md](docs/claims_report.md)**: Formal verification that all 8 research hypotheses (H1 to H8) hold strictly true.

---

## 7. Credits & Acknowledgements

We express our sincere gratitude and credit to the authors of **SecureDrive-FL** for proposing the foundational protocol and providing the high-level design of the codebase and architecture:

> **SecureDrive-FL: Joint Differential Privacy and Gradient-Aware Selective Homomorphic Encryption for Federated Driver Monitoring**  
> *Baran Can Gül, Hanuma Siddhartha Tunuguntla, Anjana Arvind Naik, Abhishek Vijay Potekar, Nasser Jazdi, and Michael Weyrich*  
> Institute of Industrial Automation and Software Engineering, University of Stuttgart & Eindhoven University of Technology (TU/e)  
> Preprint: [arXiv:2608.27108](https://arxiv.org/abs/2608.27108)

Our implementation builds directly upon their high-level architectural design—specifically the closed-loop coupling of client-side DP-SGD, Gradient-Aware Selective Homomorphic Encryption (GASHE), closed-form sensitivity thresholding ($\tau = \sigma C$), and server-side 3-$\sigma$ LossGuard validation.

---

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
