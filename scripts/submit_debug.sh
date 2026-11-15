#!/bin/bash
#SBATCH --job-name=sdfl_debug
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out

# Fail loudly on error
set -euo pipefail

echo "=========================================================="
echo "Starting Debug Job at $(date)"
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

echo "Running on $(hostname)"
echo "Using Python: $($PY --version 2>&1) at $PY"

# Run validation test suite (10 base tests + 9 extension invariant tests + 14 specification tests = 33 tests)
echo "Running pytest test suite (33 formal verification checks)..."
$PY -m pytest tests/test_securedrive.py tests/test_budget_invariants.py -v

# Run toy demo
echo "Running Phase 2 reference demo..."
$PY demo.py

echo "Job completed"
echo "=========================================================="
echo "Debug job completed successfully at $(date)"
echo "=========================================================="
