#!/usr/bin/env bash
#SBATCH --job-name=bw1-ddp
#SBATCH --partition=sixhour
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/bw1_ddp_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/bw1_ddp_%j.err

# Repro DDP world=2 + no_sync (path lsgd) SIN torchrun: la IMA da traceback Python.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF="$BASE/pytorch-cuda.sif"
OVL="$BASE/blackwell-torch.overlay"
VENV=/bb/bwvenv
cd "$BASE"
srun --export=ALL -n 2 apptainer exec --nv --overlay "$OVL":ro "$SIF" env \
  PYTHONDONTWRITEBYTECODE=1 CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 \
  "$VENV/bin/python" "$BASE/scripts/repro_ddp_nosync.py" 2>&1 | tail -60
echo "bw1-ddp done"
