#!/usr/bin/env bash
#SBATCH --job-name=bw1-mini
#SBATCH --partition=sixhour
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/bw1_mini_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/bw1_mini_%j.err

# Repro MINIMO de 1 solo rank (sin DDP) con aserciones de device = traceback Python con linea.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF="$BASE/pytorch-cuda.sif"
OVL="$BASE/blackwell-torch.overlay"
VENV=/bb/bwvenv
cd "$BASE"
apptainer exec --nv --overlay "$OVL":ro "$SIF" env PYTHONDONTWRITEBYTECODE=1 \
  CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 \
  "$VENV/bin/python" "$BASE/scripts/repro_1fwd1bwd.py" 2>&1 | tail -60
echo "bw1-mini done"
