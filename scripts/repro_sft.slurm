#!/usr/bin/env bash
#SBATCH --job-name=repro-sft
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/repro_sft_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/repro_sft_%j.err
# Repro: assert 'index out of bounds' en el batch ~1881 del SFT v3.
# CUDA_LAUNCH_BLOCKING + TORCH_USE_CUDA_DSA -> stacktrace exacto del modeling.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
export PYTHONPATH="$BASE/venv-extra:$BASE/scripts:${PYTHONPATH:-}"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
apptainer exec --nv --overlay "$BASE/blackwell-torch.overlay":ro "$BASE/pytorch-cuda.sif" \
  env PYTHONPATH="$BASE/venv-extra:$BASE/scripts:${PYTHONPATH:-}" \
  CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 \
  /bb/bwvenv/bin/python "$BASE/scripts/repro_sft_gate.py" --start-bi 1880
echo "repro rc=$?"