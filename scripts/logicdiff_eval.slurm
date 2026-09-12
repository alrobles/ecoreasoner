#!/bin/bash
#SBATCH --job-name=logicdiff-eval
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/logicdiff_eval_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/logicdiff_eval_%j.err

# Batería L0-L3 con scoring LogicDiff sobre el eval set LIMPIO
# (pairs_hard_v3_eval, seed 9999 — comparable con los NO-GO de la familia).

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07

CKPT=${CKPT:-$BASE/runs/f0-span-v3-role/checkpoint-g10000/model.pt}
CFG=${CFG:-$BASE/harness/configs/f0-span-v3-role.yaml}
HEAD=${HEAD:-$BASE/runs/logicdiff-head-v3role/head.pt}
PAIRS=${PAIRS:-$BASE/runs/pairs_hard_v3_eval}
OUT=${OUT:-$BASE/runs/logicdiff-eval-v3role}
MODES=${MODES:-base,dense,staged,staged_rev,staged_rand,payload,staged_head}

echo "=== logicdiff battery $(date -u) ==="
echo "CKPT=$CKPT HEAD=$HEAD PAIRS=$PAIRS MODES=$MODES"

apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python -u \
    "$BASE/harness/suite_smoke_logicdiff.py" \
    --ckpt "$CKPT" --config "$CFG" \
    --pairs-dir "$PAIRS" --out-dir "$OUT" \
    --modes "$MODES" --head "$HEAD" --tokenizer "$TOK" \
    --device cuda ${LIMIT:+--limit $LIMIT}

for m in ${MODES//,/ }; do
  echo "--- verdict $m ---"
  apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python \
      "$BASE/scripts/verdict.py" --battery "$OUT/$m" \
      --out "$OUT/$m/verdict.json" 2>/dev/null || true
done

echo "=== DONE $(date -u) ==="
