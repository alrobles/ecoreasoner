#!/usr/bin/env bash
#SBATCH --job-name=smoke-lora
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/smoke_lora_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/smoke_lora_%j.err
# Smoke post-SFT: base + LoRA final -> tool calls L1 validas?
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
export PYTHONPATH="$BASE/venv-extra:${PYTHONPATH:-}"
apptainer exec --nv --overlay "$BASE/blackwell-torch.overlay":ro "$BASE/pytorch-cuda.sif" \
  /bb/bwvenv/bin/python "$BASE/scripts/smoke_moe_lora.py" \
  --n 40 --gen-length 160 --steps 125 \
  --out "$BASE/data/l1/smoke_moe_lora.json"
echo "smoke-lora done rc=$?"