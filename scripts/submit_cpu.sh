#!/bin/bash
#SBATCH --job-name=sdfl_cpu
#SBATCH --partition=gpu
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out

# Fail loudly on error
set -euo pipefail

echo "=========================================================="
echo "Starting CPU Job at $(date)"
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

# Run experiment matrix across 9 baselines, 3 conditions, 3 seeds
ROUNDS=${1:-120}
shift || true
EXTRA_ARGS="$@"
echo "Running experiment matrix with $ROUNDS rounds (extra flags: ${EXTRA_ARGS:-none})..."
$PY -m securedrive_fl.experiments --rounds "$ROUNDS" --seeds 42 123 456 --results-dir results $EXTRA_ARGS

# Generate deliverables
echo "Generating deliverables (Table I, Table II, Fig. 2)..."
$PY -m securedrive_fl.make_report

echo "Job completed"
echo "=========================================================="
echo "CPU Job completed successfully at $(date)"
echo "=========================================================="
