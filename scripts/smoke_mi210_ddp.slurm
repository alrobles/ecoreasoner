#!/bin/bash
#SBATCH --job-name=smoke-mi210ddp
#SBATCH --partition=sixhour
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:mi210:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/smoke_mi210_ddp_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/smoke_mi210_ddp_%j.err
# Spike ROCm fase 2: DDP 3-rank intra-nodo (RCCL) de v5-1b via venv_rocm.
set -euo pipefail
BASE=/beegfs/a474r867
export ROCM_PATH="/kuhpc/sw/rocm/6.4.3"
export PATH="$BASE/venv_rocm/bin:${ROCM_PATH}/bin:$PATH"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=${MASTER_PORT:-29518}
echo "[$(hostname)] rocm=$ROCM_PATH"
torchrun --nproc_per_node=3 "$BASE/ecoreasoner/scripts/smoke_mi210_ddp.py" ${EXTRA:-}
