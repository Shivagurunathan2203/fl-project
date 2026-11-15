# EECS Cluster Deployment & User Guide: SecureDrive-FL & Budget Extension

This guide provides comprehensive, step-by-step instructions for running the corrected **SecureDrive-FL** framework and the **Compute-Budgeted Selective Encryption Extension** (`spec_budget.md`) on the **EECS Department HPC Cluster** (`172.30.2.64`), tailored specifically to your cluster configuration, file system layout, Kaggle API setup, and Slurm batch environment.

---

## 1. Project Directory Structure

```text
sdfl_code/
├── securedrive_fl/                  # Core SecureDrive-FL framework
│   ├── config.py                    # Base configuration dataclass (DP noise z, tau, sigma0)
│   ├── he.py                        # HEBackend Protocol, MockHE, and TenSealCKKS adapter
│   ├── gashe.py                     # GASHE 3-tier threshold mask, Payload, make_payload
│   ├── aggregate.py                 # Server-side homomorphic aggregation (no server decryption)
│   ├── attacks.py                   # Blanchard MitM channel tampering & backdoor triggers
│   ├── guard.py                     # 3-sigma LossGuard (filters loss spikes, rolls back state)
│   ├── privacy.py                   # Rényi Differential Privacy (RDP) accountant
│   ├── train.py                     # Client-side DP-SGD (clipping C=1.2, noise z*C), Eq. 13 backdoor
│   ├── model.py                     # MobileNetV2-Tiny (2,236,682 params), CPU/GPU auto-detection
│   ├── data.py                      # Calibrated 10-class toy federation & AUC/State Farm loader
│   ├── baselines.py                 # 9 baseline configurations & BatchCrypt 16-bit quantization
│   ├── metrics.py                   # Multiclass Confusion Matrix, ASR, F1, and Precision
│   ├── run_scenario.py              # FL round simulation loop with 1-round delay
│   ├── experiments.py               # Experiment matrix runner with RSS memory tracker
│   ├── make_report.py               # Generates Table I, Table II, and Figure 2
│   └── budget/                      # Compute-Budgeted Selective Encryption Package (Extension)
│       ├── __init__.py              # Extension public API exports
│       ├── config.py                # BudgetConfig, CapacityConfig, AvailabilityConfig
│       ├── model.py                 # BudgetModel (heterogeneous tiers, Markov/uniform availability)
│       ├── codec.py                 # Budgeted GASHE chunk codec (post-noise energy ranking)
│       ├── aggregate.py             # Server-side chunked homomorphic aggregator
│       ├── time_model.py            # LatencyModel (straggler wait cost, deadline modeling)
│       ├── scenarios.py             # Scenario definitions (H1 hetero_tiers, S1 sweep, S2 bursty)
│       ├── strategies.py            # Strategy execution loop (sd_budget, sd_wait, sd_exclude)
│       └── inversion.py             # Reconstruction audit (DLG gradient inversion leakage)
├── scripts/                         # Slurm batch submission & utility scripts
│   ├── download_auc_dataset.py      # Automated AUC dataset downloader via Kaggle API
│   ├── calibrate_he_cost.py         # Calibrates TenSEAL CKKS cryptographic timings
│   ├── run_grid_ext.py              # Extension grid runner (H1, S1 sweeps, S2 bursty)
│   ├── run_audit.py                 # DLG gradient inversion audit runner across budgets f
│   ├── make_tables_ext.py           # Generates Table I-ext and Table II-ext (MD/CSV)
│   ├── make_figures_ext.py          # Generates Figures 2-ext, 3, 4, 5, 6 (PNG/PDF)
│   ├── make_claims_report.py        # Generates claims_report.md evaluating 8 research claims
│   ├── submit_debug.sh              # Quick validation script (runs pytest & demo on 'gpu')
│   ├── submit_gpu.sh                # Base paper reproduction Slurm script (81 runs)
│   ├── submit_extension.sh          # Extension Slurm script (grid, audit, tables, figures)
│   └── submit_cpu.sh                # Production script on CPU partition
├── tests/
│   ├── test_securedrive.py          # 10 formal verification tests from base paper spec
│   └── test_budget_invariants.py    # 9 formal invariants (I1 to I9) from extension spec
├── data/                            # Distracted driver datasets
│   ├── auc/                         # AUC Distracted Driver dataset directory
│   ├── statefarm/                   # State Farm Distracted Driver directory
│   └── README.md                    # Dataset instructions
├── results/                         # Cached JSON results for 9 base configurations
├── results_ext/                     # Cached JSON/CSV results and calibration for extension
├── table_1.md / table_1.csv         # Base Table I: Final Accuracy, ASR, F1, Precision
├── table_2.md / table_2.csv         # Base Table II: Wall-Clock, Relative Overhead, Peak RSS
├── fig2.png / fig2.pdf              # Base Figure 2: Convergence curves under MitM and Poison
├── table_1_ext.md / table_1_ext.csv # Extension Table I-ext (H1 heterogeneous budgets)
├── table_2_ext.md / table_2_ext.csv # Extension Table II-ext (Timing, straggler, byte metrics)
├── fig2_ext.* / fig3_sweep.*        # Extension Figures 2-ext, 3 (sweep), 4 (leakage), 5, 6
├── claims_report.md                 # Automated claims verdict report (8 research claims)
├── DEVIATIONS_EXT.md                # Log of all extension architectural decisions
├── REPORT_EXT.md                    # Full analysis of extension findings and paper claims
├── RESULTS.md                       # Comprehensive report of base paper reproduction
├── requirements.txt                 # Dependencies for conda environment
├── demo.py                          # Standalone demo running scenarios
└── guide.md                         # This deployment guide
```

---

## 2. Server Access & Storage Guidelines

### Cluster Credentials & Details
- **Server IP**: `172.30.2.64`
- **Username**: `group11`
- **Server Home Path**: `/driveB/FL/group11` (or `~`)
- **Remote Project Directory**: `/driveB/FL/group11/sdfl_code` (or `~/sdfl_code`)
- **Local Project Directory**: `C:\Users\kusha\Downloads\sem7\FL\Project\sdfl_code`

### Storage Quotas & Limits
All user files reside under `/driveB/FL/group11`. Storage quotas are enforced by role:

| User Role | Storage Quota | Priority Level |
| :--- | :---: | :---: |
| Faculty | 50 GB | Highest |
| PhD / Postdoc / RA | 50 GB | High |
| MS Students | 20 GB | Medium |
| BS Students (Group 11) | 20 GB | Lowest |

> [!IMPORTANT]
> The department is not responsible for data loss due to hardware failure. Always back up important results to your local computer. Check disk usage frequently with:
> ```bash
> du -sh *      # Check disk usage of files/folders in current directory
> df -h         # Check total disk quota and server storage
> ```

---

## 3. Kaggle API Setup & Downloading the State Farm Driver Dataset

Configure your Kaggle API credentials via environment variables (`KAGGLE_USERNAME` and `KAGGLE_KEY`) or in `~/.kaggle/kaggle.json`:
```json
{
  "username": "<your_kaggle_username>",
  "key": "<your_kaggle_key>"
}
```

We use the **State Farm Distracted Driver Detection** benchmark dataset. It contains the exact same 10 distracted-driving classes (`c0`–`c9`) as AUC, requires **no academic agreements or passwords**, and downloads cleanly via the Kaggle API.

### Run State Farm Dataset Download on the Server:
```bash
python scripts/download_statefarm_dataset.py
```

*What this script does*:
1. Reads credentials from environment variables or `~/.kaggle/kaggle.json`.
2. Cleans up any leftover encrypted AUC zip archives in `data/auc/` to reclaim ~2.2 GB of disk quota.
3. Downloads the ground truth driver mapping `driver_imgs_list.csv` (800 KB, mapping 26 real drivers across classes `c0`–`c9`).
4. Downloads the complete 10-class dataset archive `mdriyadulhasan/distracted-driver-dataset` (906 MB).
5. Extracts all 10 classes (`c0` through `c9`) into `data/statefarm/`.
6. Automatically deletes the downloaded `.zip` file after extraction to keep disk usage well under your 20 GB quota.

*Expected Output*:
```text
[Info] Authenticated as Kaggle user: kushalgangadhar
[Info] Connecting to Kaggle: driver_imgs_list.csv metadata...
[Info] Connecting to Kaggle: State Farm Distracted Driver (mdriyadulhasan/distracted-driver-dataset)...
[Info] Content-Length: 949294931 bytes (~905.3 MB)
Downloading: 905.3/905.3 MB [100.0%]
[Info] Download complete!
[Info] Extracting distracted_driver_dataset.zip to data/statefarm...
[Info] Removed archive distracted_driver_dataset.zip to free disk space.
======================================================================
[Success] State Farm Distracted Driver dataset is ready in data/statefarm!
          Total images: 22424
          Classes: 10 (c0: safe driving, c1..c9: various distractions)
          Metadata: data/statefarm/driver_imgs_list.csv
======================================================================
```

---

## 4. Step-by-Step Instructions: EECS Cluster Workflow

### Step 1: Transfer Code to the Server

> [!WARNING]
> `scp -r` **does NOT automatically delete** older files on the remote server. It only overwrites matching files. Stale code, old log files, or old cached results from prior runs will remain on the cluster unless removed.
>
> To ensure a clean deployment and avoid leaving stale files or wasting disk quota:

Open **PowerShell** on your local Windows PC (from `C:\Users\kusha\Downloads\sem7\FL\Project`):

```powershell
# OPTION 1 (Incremental Update - recommended if you want to keep data/ on the server):
# Syncs only the updated python scripts and package files:
scp -r sdfl_code/scripts sdfl_code/securedrive_fl sdfl_code/guide.md group11@172.30.2.64:~/sdfl_code/

# OPTION 2 (Fresh Full Clean Copy):
# Removes older files and re-copies the full folder:
ssh group11@172.30.2.64 "rm -rf ~/sdfl_code"
cd C:\Users\kusha\Downloads\sem7\FL\Project
scp -r sdfl_code group11@172.30.2.64:~/
```

---

### Step 2: Connect via SSH & Setup Environment

Connect to the login node:
```powershell
ssh group11@172.30.2.64
```

Navigate to the project folder:
```bash
cd ~/sdfl_code
```

Set up the Conda environment using Environment Modules:
```bash
# 1. Clear pre-loaded modules
module purge

# 2. Load Miniconda module
module load conda/miniconda3

# 3. Accept Anaconda Terms of Service (if prompted)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 4. Create the dedicated Python 3.10 environment (if not already created)
conda create -n sdfl_env python=3.10 -y

# 5. Reload shell configuration and activate
source ~/.bashrc
conda activate sdfl_env

# 6. Install required packages inside active environment
pip install -r requirements.txt
```

---

### Step 3: Validate Code and Invariants on Slurm

Submit the validation job to verify all 33 unit tests (10 base tests + 9 extension invariant tests + 14 specification tests) and demo execution:

```bash
sbatch scripts/submit_debug.sh
```

Monitor job status:
```bash
squeue -u $USER
```

Once finished (~30–45 seconds), inspect the output log:
```bash
cat sdfl_debug_*.out
```
*Expected Output*:
- `10 passed in ...s` (Base paper tests: mask split, aggregation equivalence, tau/z calibration, RDP accounting, LossGuard, rollback, sorting, server cannot decrypt, MobileNet flattening).
- `9 passed in ...s` (Extension invariant tests: I1 unlimited budget equivalence, I2 zero budget, I3 exactness across modes, I4 clean equivalence, I5 monotone protection, I6 backend parity, I7 budget model, I8 server cannot decrypt, I9 exclude strategy).
- Demo qualitative scenario outputs.

---

### Step 4: Recreate the Base Paper Experiments (Table I, Table II, Fig. 2)

The corrected code runs the full 81-scenario experiment matrix across 9 baseline configurations, 3 attack conditions, and 3 random seeds.

#### Option A: Running on Toy Calibrated Dataset
```bash
sbatch scripts/submit_gpu.sh 25
```
*(Runs 25 FL rounds per scenario; to run 120 rounds as in the paper, pass `sbatch scripts/submit_gpu.sh 120`).*

#### Option B: Running on the Real AUC Distracted Driver Dataset
First download the dataset (if not already downloaded):
```bash
python scripts/download_auc_dataset.py
```
Then submit the GPU job with the real data flag:
```bash
python -m securedrive_fl.experiments --rounds 25 --seeds 42 123 456 --real-data --data-dir data --results-dir results
python -m securedrive_fl.make_report
```

*Generated Deliverables*:
- `table_1.md` / `table_1.csv`
- `table_2.md` / `table_2.csv`
- `fig2.png` / `fig2.pdf`
- `RESULTS.md`

---

### Step 5: Recreate the Extension (Budgeted Selective Encryption)

The extension experiments evaluate compute budgets, straggler latencies, client exclusion, and gradient inversion leakage.

Submit the full extension pipeline via Slurm:
```bash
sbatch scripts/submit_extension.sh 25
```

Or run interactively step-by-step:
```bash
# 1. Calibrate TenSEAL CKKS cryptographic operation timings
python scripts/calibrate_he_cost.py

# 2. Run Reconstruction Audit (DLG gradient inversion leakage across budgets f)
python scripts/run_audit.py

# 3. Run Extension Experiment Grid (H1 headline, S1 sweep, S2 bursty)
python scripts/run_grid_ext.py --rounds 25 --seeds 42 123 456

# 4. Generate Extension Tables, Figures, and Claims Report
python scripts/make_tables_ext.py
python scripts/make_figures_ext.py
python scripts/make_claims_report.py
```

*Generated Deliverables*:
- `table_1_ext.md` / `table_1_ext.csv` (Table I-ext: Accuracy and ASR under H1)
- `table_2_ext.md` / `table_2_ext.csv` (Table II-ext: Timing, straggler, and upload bytes)
- `fig2_ext.png` / `fig2_ext.pdf` (Convergence trajectories under MitM and Poison)
- `fig3_sweep.png` / `fig3_sweep.pdf` (Budget sweep S1: accuracy and protected energy vs f)
- `fig4_leakage.png` / `fig4_leakage.pdf` (Reconstruction leakage PSNR vs budget f)
- `fig5_cost_benefit.png` / `fig5_cost_benefit.pdf` (Accuracy vs simulated execution time)
- `fig6_pertier.png` / `fig6_pertier.pdf` (Per-tier protected gradient energy bar chart)
- `claims_report.md` (Verdicts on all 8 research hypotheses)
- `DEVIATIONS_EXT.md` and `REPORT_EXT.md`

---

### Step 6: Download Deliverables Back to Local Computer

Once Slurm jobs complete, open **PowerShell** on your local machine (`C:\Users\kusha\Downloads\sem7\FL\Project\sdfl_code`):

```powershell
# 1. Download Base Paper Deliverables
scp group11@172.30.2.64:~/sdfl_code/table_1.* ./
scp group11@172.30.2.64:~/sdfl_code/table_2.* ./
scp group11@172.30.2.64:~/sdfl_code/fig2.* ./
scp group11@172.30.2.64:~/sdfl_code/RESULTS.md ./

# 2. Download Extension Deliverables
scp group11@172.30.2.64:~/sdfl_code/table_*_ext.* ./
scp group11@172.30.2.64:~/sdfl_code/fig*ext.* ./
scp group11@172.30.2.64:~/sdfl_code/fig3_sweep.* ./
scp group11@172.30.2.64:~/sdfl_code/fig4_leakage.* ./
scp group11@172.30.2.64:~/sdfl_code/fig5_cost_benefit.* ./
scp group11@172.30.2.64:~/sdfl_code/fig6_pertier.* ./
scp group11@172.30.2.64:~/sdfl_code/claims_report.md ./
scp group11@172.30.2.64:~/sdfl_code/REPORT_EXT.md ./
scp group11@172.30.2.64:~/sdfl_code/DEVIATIONS_EXT.md ./

# 3. Download Full Results Folders
scp -r group11@172.30.2.64:~/sdfl_code/results/ ./
scp -r group11@172.30.2.64:~/sdfl_code/results_ext/ ./
```

---

## 5. Slurm Monitoring & Command Cheat Sheet

| Task | Command | Description |
| :--- | :--- | :--- |
| View your active jobs | `squeue -u $USER` | Shows queued and running jobs |
| Live console streaming | `tail -f sdfl_*.out` | Follow console logs in real time |
| Inspect job details | `scontrol show job <jobid>` | Allocated nodes, time limit, memory |
| Check partition status | `sinfo` | Partition availability (gpu, cpu) |
| Inspect GPU resources | `nvidia-smi` | GPU memory and compute utilization |
| Cancel a specific job | `scancel <jobid>` | Kills job with given ID |
| Cancel all your jobs | `scancel -u $USER` | Kills all active/pending jobs for your user |
| Job history & usage | `sacct -u $USER` | CPU/memory usage and exit codes |

---

## 6. Daily Quick Reference Card

```bash
# ─── CLEAN DEPLOYMENT (Run from Local PC) ───────────────
ssh group11@172.30.2.64 "rm -rf ~/sdfl_code"
scp -r sdfl_code group11@172.30.2.64:~/

# ─── CONNECT (Local PC) ─────────────────────────────────
ssh group11@172.30.2.64

# ─── SETUP ENVIRONMENT (Server) ─────────────────────────
module purge
module load conda/miniconda3
conda activate sdfl_env

# ─── DOWNLOAD DATASETS (Server) ─────────────────────────
python scripts/download_auc_dataset.py     # Distracted Driver (AUC)

# ─── SUBMIT SLURM JOBS (Server) ─────────────────────────
sbatch scripts/submit_debug.sh         # Run 33 verification tests & demo
sbatch scripts/submit_gpu.sh 25        # Run Base Paper reproduction (State Farm/AUC)
sbatch scripts/submit_extension.sh 25  # Run Extension grid (State Farm/AUC)

# ─── MONITORING (Server) ────────────────────────────────
squeue -u $USER
tail -f sdfl_gpu_*.out
tail -f sdfl_ext_*.out

# ─── TRANSFER DELIVERABLES (Local PC) ───────────────────
scp group11@172.30.2.64:~/sdfl_code/table_*.md ./
scp group11@172.30.2.64:~/sdfl_code/table_*.csv ./
scp group11@172.30.2.64:~/sdfl_code/fig*.* ./
scp group11@172.30.2.64:~/sdfl_code/claims_report.md ./
scp group11@172.30.2.64:~/sdfl_code/REPORT_EXT.md ./
```
