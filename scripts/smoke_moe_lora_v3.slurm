#!/usr/bin/env bash
#SBATCH --job-name=smoke-lora-v3
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/smoke_lora_v3_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/smoke_lora_v3_%j.err
# Smoke post-SFT v3: base + lora-final v3 -> tool calls L1 validas?
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
export PYTHONPATH="$BASE/venv-extra:${PYTHONPATH:-}"
apptainer exec --nv --overlay "$BASE/blackwell-torch.overlay":ro "$BASE/pytorch-cuda.sif" \
  /bb/bwvenv/bin/python "$BASE/scripts/smoke_moe_lora.py" \
  --lora "$BASE/outputs/sft_moe_v3/lora-final/lora.pt" \
  --n 40 --gen-length 160 --steps 125 \
  --out "$BASE/data/l1/smoke_moe_lora_v3.json"
echo "smoke-lora-v3 done rc=$?"