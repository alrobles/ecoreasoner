#!/usr/bin/env bash
#SBATCH --job-name=sft-moe
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=100G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/sft_moe_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/sft_moe_%j.err
# SFT del LLaDA-MoE-7B-A1B-Instruct con trazas L1 (tool-calling eco).
# Receta paper LLaDA 2.3: 3 epochs, lr 2.5e-5 (warmup 50), wd 0.1, LoRA r64.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
export PYTHONPATH="$BASE/venv-extra:${PYTHONPATH:-}"
apptainer exec --nv --overlay "$BASE/blackwell-torch.overlay":ro "$BASE/pytorch-cuda.sif" \
  /bb/bwvenv/bin/python "$BASE/scripts/sft_lladamoe.py" \
  --batch 8 --max-len 1536 --rank 64 --epochs 3 \
  --pairs "$BASE/data/l1/sft_moe_pairs.jsonl" \
  --out "$BASE/outputs/sft_moe"
echo "sft done rc=$?"