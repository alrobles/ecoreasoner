#!/bin/bash
#SBATCH --partition=sixhour
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gres=gpu:mi210:2
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/evalA_mini-%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/evalA_mini-%j.err
#SBATCH --job-name=eco-evalA-mini
set -u
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/qwen35-rocm-v2.sif
MODEL=$BASE/models/Qwen3.5-35B-A3B
OUT=$BASE/outputs

echo "=== MINI EVAL $(date -u) host=$(hostname) ==="
# 1) check torch/GPU dentro del SIF
apptainer exec --nv $SIF python3 -c "import torch; print('ROCm:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())" 2>&1

# 2) mini eval: 1 replica, 2 prompts
head -2 $BASE/data/prompts_toolcall_canonical.jsonl > /tmp/prompts_mini.jsonl
head -2 /beegfs/a474r867/ecoreasoner/data/sci_v1_prompts.jsonl > /tmp/mem_mini.jsonl

apptainer exec --nv $SIF \
  python3 -u /beegfs/a474r867/ecoreasoner/scripts/eval_baseline_A.py \
  --base $MODEL \
  --adapters "$OUT/m1/final" \
  --prompts /tmp/prompts_mini.jsonl \
  --mem-check /tmp/mem_mini.jsonl \
  --max-new-tokens 64 \
  --out /tmp/evalA_mini.json 2>&1 | tail -30

echo "=== fin mini eval rc=$? ==="
ls -la /tmp/evalA_mini.json 2>/dev/null && echo "JSON_OK"