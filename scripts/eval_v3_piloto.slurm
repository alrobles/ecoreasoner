#!/bin/bash
#SBATCH --job-name=eval-piloto-v3
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/eval_piloto_v3_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/eval_piloto_v3_%j.err

set -e

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
CKPT=$BASE/runs/f0-span-esqueleto-v2-piloto/checkpoint-g10000/model.pt
CONFIG=$BASE/harness/configs/f0-span-esqueleto-v2.yaml
PAIRS_DIR=$BASE/runs/pairs_hard_v3
OUT_DIR=$BASE/runs/f0-span-esqueleto-v2-piloto/battery_v3

apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python \
    "$BASE/harness/suite_smoke_v2.py" \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --pairs-dir "$PAIRS_DIR" \
    --out-dir "$OUT_DIR"

apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python \
    "$BASE/scripts/verdict.py" --battery "$OUT_DIR" \
    --out "$BASE/runs/f0-span-esqueleto-v2-piloto/battery_v3_verdict.json"

echo "[done] eval v3 escrita en $OUT_DIR"
