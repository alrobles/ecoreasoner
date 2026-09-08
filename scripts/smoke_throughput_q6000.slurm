#!/usr/bin/env bash
#SBATCH --job-name=thrpt-q6000
#SBATCH --partition=sixhour
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=48G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/thrpt-q6000_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/thrpt-q6000_%j.err
# =============================================================================
# Smoke de THROUGHPUT pro6000 (cu128, 
# NO soporta sm_120). Mismo metodo que L40/A100/Q6000: 200 steps del micro
# dense, cronometrando SOLO el entrenamiento (excluye la carga del cache).
# =============================================================================
set -eo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
OUT=$BASE/runs/thrpt_q6000
mkdir -p "$OUT"
echo "=== GPU detect $(hostname) $(date -u) ==="
apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python -c "
import torch
p = torch.cuda.get_device_properties(0)
print('GPU:', p.name, '| VRAM GB:', round(p.total_memory/1e9), '| torch:', torch.__version__)
"
echo "=== entrenando 200 steps del micro-modelo (batch 8, seq 768) ==="
apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python "$BASE/scripts/train_mdlm_moe.py" \
    --data_cache "$BASE/data/train_ids_v7_clean.npy" \
    --data "$BASE/data/train_ids_v7_clean.npy" \
    --vocab 126080 --hidden 512 --layers 8 --heads 8 --ff_mult 4 \
    --n_experts 1 --expert_k 1 --seq_len 768 \
    --batch_size 8 --grad_accum 2 --mask_p 0.15 \
    --mask_type random --span_len 64 \
    --lr 2e-4 --warmup 50 --max_steps 200 \
    --output "$OUT"
echo "=== DONE $(date -u) ==="