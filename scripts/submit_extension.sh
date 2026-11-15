#!/bin/bash
#SBATCH --job-name=sdfl_ext
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out

# Fail loudly on error
set -euo pipefail

echo "=========================================================="
echo "Starting SecureDrive-FL Extension Job on $(hostname) at $(date)"
echo "=========================================================="

# Activate Conda environment
source /opt/conda/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate sdfl_env 2>/dev/null || true

# Explicitly resolve Python from sdfl_env
if [ -x "$HOME/.conda/envs/sdfl_env/bin/python" ]; then
    PY="$HOME/.conda/envs/sdfl_env/bin/python"
elif [ -x "/opt/conda/miniconda3/envs/sdfl_env/bin/python" ]; then
    PY="/opt/conda/miniconda3/envs/sdfl_env/bin/python"
else
    PY="$(which python3)"
fi

echo "Using Python: $($PY --version 2>&1) at $PY"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-'Not set'}"

# 1. Step 1: Calibrate TenSEAL CKKS Cryptographic Timings
echo "--- Step 1: Calibrating HE Cryptographic Timings ---"
$PY scripts/calibrate_he_cost.py

# 2. Step 2: Run Reconstruction Audit (Gradient Inversion DLG)
echo "--- Step 2: Running Gradient Inversion Leakage Audit ---"
$PY scripts/run_audit.py

# 3. Step 3: Run Extension Grid across H1, S1, S2 Scenarios
ROUNDS=${1:-120}
shift || true
EXTRA_ARGS="$@"
echo "--- Step 3: Running Extension Experiment Grid ($ROUNDS rounds, extra flags: ${EXTRA_ARGS:-none}) ---"
$PY scripts/run_grid_ext.py --rounds "$ROUNDS" --seeds 42 123 456 $EXTRA_ARGS

# 4. Step 4: Generate Extension Deliverables (Tables, Figures, Claims, Reports)
echo "--- Step 4: Generating Tables, Figures, and Reports ---"
$PY scripts/make_tables_ext.py
$PY scripts/make_figures_ext.py
$PY scripts/make_claims_report.py
$PY scripts/make_results_md.py
$PY scripts/make_report_ext_md.py

echo "=========================================================="
echo "Extension Job completed successfully at $(date)"
echo "=========================================================="
