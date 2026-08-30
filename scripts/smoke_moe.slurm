#!/usr/bin/env bash
#SBATCH --job-name=smoke-moe
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/smoke_moe_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/smoke_moe_%j.err

# Smoke baseline del base LLaDA-MoE-7B-A1B-Instruct (tool-calling ya entrenado,
# Apache 2.0) sobre prompts eco L1. GPU libre (training bw3 usa 2 de un nodo).
# Mide si el Instruct YA emite tool calls JSON validas SIN fine-tune.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
OVL="$BASE/blackwell-torch.overlay"
SIF="$BASE/pytorch-cuda.sif"
apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python \
  "$BASE/scripts/smoke_moe.py" \
  --n 40 --gen-length 180 --steps 128 \
  --out "$BASE/data/l1/smoke_moe_baseline.json"
echo "smoke-moe done rc=$?"