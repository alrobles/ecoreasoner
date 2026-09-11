#!/bin/env bash
#SBATCH --job-name=f0-v3-contrastive
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/f0_v3_contrastive_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/f0_v3_contrastive_%j.err

set -e

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay

CKPT="${CKPT:-/beegfs/a474r867/ecoreasoner/runs/f0-span-esqueleto-v2-piloto/checkpoint-g10000/model.pt}"
CONFIG="${CONFIG:-/beegfs/a474r867/ecoreasoner/harness/configs/f0-span-v3-contrastive.yaml}"
PAIRS="${PAIRS:-/beegfs/a474r867/ecoreasoner/runs/pairs_hard_v3/pairs_L3.jsonl}"
DATA_CACHE="${DATA_CACHE:-/beegfs/a474r867/ecoreasoner/data/train_ids_skeleton_v2.npz}"
OUTPUT="${OUTPUT:-/beegfs/a474r867/ecoreasoner/runs/f0-span-v3-contrastive}"

MAX_STEPS="${MAX_STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
ALPHA="${ALPHA:-0.2}"
MARGIN="${MARGIN:-0.0}"
SEED="${SEED:-7331}"

apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python -c "
import torch
print('GPUs:', torch.cuda.device_count())
assert torch.cuda.device_count() >= 1, 'sin GPU'
" || { echo 'GPU detect FAIL'; exit 1; }

echo "=== Lanzando V3 contrastive sobre $CKPT (steps=$MAX_STEPS, lr=$LR, alpha=$ALPHA) $(date -u) ==="

apptainer exec --nv --overlay "$OVL":ro "$SIF" \
    /bb/bwvenv/bin/python "$BASE/scripts/train_mdlm_moe_v3_contrastive.py" \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --pairs "$PAIRS" \
    --data_cache "$DATA_CACHE" \
    --output "$OUTPUT" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --alpha "$ALPHA" \
    --margin "$MARGIN" \
    --seed "$SEED"

echo "=== Evaluando bateria L0-L3 ==="
CKPT_FINAL=$(ls -d "$OUTPUT"/checkpoint-g* | sort -V | tail -1)/model.pt

apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python \
    "$BASE/harness/suite_smoke_v2.py" \
    --ckpt "$CKPT_FINAL" --config "$CONFIG" \
    --pairs-dir /beegfs/a474r867/ecoreasoner/runs/pairs_hard_v3 \
    --out-dir "$OUTPUT/battery" --device cuda

apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python \
    "$BASE/scripts/verdict.py" --battery "$OUTPUT/battery" \
    --out "$OUTPUT/battery_verdict.json"

echo "=== V3 contrastive DONE $(date -u) ==="
