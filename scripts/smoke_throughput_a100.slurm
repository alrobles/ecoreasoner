#!/usr/bin/env bash
#SBATCH --job-name=thrpt-a100
#SBATCH --partition=sixhour
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/thrpt-a100_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/thrpt-a100_%j.err
# =============================================================================
# Smoke de THROUGHPUT A100 (F1-hetero, 2026-09-07): mide tok/s REALES del
# micro-modelo dense 50-100M (el mismo del F0 sweep) entrenando 200 steps con
# el trainer real (train_mdlm_moe.py) y el SIF base cu121.
# =============================================================================
set -eo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OUT=$BASE/runs/thrpt_a100
mkdir -p "$OUT"
echo "=== GPU detect $(hostname) $(date -u) ==="
apptainer exec --nv "$SIF" python3 -c "
import torch
p = torch.cuda.get_device_properties(0)
print('GPU:', p.name, '| VRAM GB:', round(p.total_memory/1e9), '| torch:', torch.__version__)
"
echo "=== entrenando 200 steps del micro-modelo (batch 8, seq 768) ==="
apptainer exec --nv "$SIF" python3 "$BASE/scripts/train_mdlm_moe.py" \
    --data_cache "$BASE/data/train_ids_v7_clean.npy" \
    --data "$BASE/data/train_ids_v7_clean.npy" \
    --vocab 126080 --hidden 512 --layers 8 --heads 8 --ff_mult 4 \
    --n_experts 1 --expert_k 1 --seq_len 768 \
    --batch_size 8 --grad_accum 2 --mask_p 0.15 \
    --mask_type random --span_len 64 \
    --lr 2e-4 --warmup 50 --max_steps 200 \
    --output "$OUT"
echo "=== DONE $(date -u) ==="