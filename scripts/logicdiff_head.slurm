#!/bin/bash
#SBATCH --job-name=logicdiff-head
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/logicdiff_head_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/logicdiff_head_%j.err

# Entrena la cabeza de roles (~4M) sobre hidden states CONGELADOS del dLLM.
# LogicDiff Fase A: el backbone no se toca (auditable).

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay

CKPT=${CKPT:-$BASE/runs/f0-span-v3-role/checkpoint-g10000/model.pt}
CFG=${CFG:-$BASE/harness/configs/f0-span-v3-role.yaml}
DATA=${DATA:-$BASE/data/logicdiff/role_train.pt}
OUT=${OUT:-$BASE/runs/logicdiff-head-v3role}
STEPS=${STEPS:-3000}

echo "=== logicdiff head train $(date -u) ==="
echo "CKPT=$CKPT  CFG=$CFG  DATA=$DATA  OUT=$OUT  STEPS=$STEPS"

apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python -u \
    "$BASE/scripts/train_logicdiff_head.py" \
    --ckpt "$CKPT" --config "$CFG" --data "$DATA" --out "$OUT" \
    --steps "$STEPS" --batch-size 16 --head-hidden 1536 \
    --device cuda

echo "=== DONE $(date -u) ==="
