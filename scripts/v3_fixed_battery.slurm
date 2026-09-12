#!/bin/bash
#SBATCH --job-name=v3-fixed-battery
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/v3_fixed_battery_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/v3_fixed_battery_%j.err

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
CKPT=$BASE/runs/f0-span-v3-role/checkpoint-g10000/model.pt
CFG=$BASE/harness/configs/f0-span-v3-role.yaml
OUT=$BASE/runs/f0-span-v3-role/battery_eval

echo "=== battery LIMPIA V3.2 role (eval set seed 9999) sobre $CKPT $(date -u) ==="

apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python \
    "$BASE/harness/suite_smoke_v2.py" \
    --ckpt "$CKPT" --config "$CFG" \
    --pairs-dir "$BASE/runs/pairs_hard_v3_eval" \
    --out-dir "$OUT" --device cuda

echo "=== verdict ==="
apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python \
    "$BASE/scripts/verdict.py" --battery "$OUT" \
    --out "$BASE/runs/f0-span-v3-contrastive-fixed/battery_eval_verdict.json"

echo "=== DONE $(date -u) ==="