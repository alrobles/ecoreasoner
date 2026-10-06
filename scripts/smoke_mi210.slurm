#!/bin/bash
#SBATCH --job-name=smoke-mi210
#SBATCH --partition=sixhour
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:mi210:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/smoke_mi210_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/smoke_mi210_%j.err
# Spike ROCm: MdLMMoE v5-1b en 1xMI210 via venv_rocm (sin apptainer).
set -euo pipefail
BASE=/beegfs/a474r867
export ROCM_PATH="/kuhpc/sw/rocm/6.4.3"
export PATH="$BASE/venv_rocm/bin:${ROCM_PATH}/bin:$PATH"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
echo "[$(hostname)] rocm=$ROCM_PATH"
nvidia-smi 2>/dev/null || rocm-smi --showproductname 2>/dev/null || true
python3 "$BASE/ecoreasoner/scripts/smoke_mi210.py" ${EXTRA:-}
