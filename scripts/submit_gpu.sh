#!/bin/bash
#SBATCH --job-name=sdfl_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out

# Fail loudly on error
set -euo pipefail

echo "=========================================================="
echo "Starting GPU Production Job at $(date)"
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
echo "GPU: ${CUDA_VISIBLE_DEVICES:-'Not set'}"

# Check GPU status (continue if driver needs reboot)
nvidia-smi || echo "Warning: NVIDIA driver mismatch on host, PyTorch will automatically use CPU if needed."

# Run experiment matrix across 9 baselines, 3 conditions, 3 seeds
ROUNDS=${1:-120}
shift || true
EXTRA_ARGS="$@"
echo "Running experiment matrix with $ROUNDS rounds (extra flags: ${EXTRA_ARGS:-none})..."
$PY -m securedrive_fl.experiments --rounds "$ROUNDS" --seeds 42 123 456 --results-dir results $EXTRA_ARGS

# Generate Table I, Table II, and Figure 2
echo "Generating deliverables (Table I, Table II, Fig. 2)..."
$PY -m securedrive_fl.make_report

echo "Job completed"
echo "=========================================================="
echo "GPU Job completed successfully at $(date)"
echo "=========================================================="
