#!/usr/bin/env bash
#SBATCH --job-name=holdout
#SBATCH --partition=sixhour
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/holdout_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/holdout_%j.err
# Eval de UN run sobre el holdout pairs_hard_v4 (Uso: RUN=xxx sbatch --export=ALL)
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
VENV=/bb/bwvenv
RUN=${RUN:?falta RUN}
PAIRS=${PAIRS:-$BASE/runs/pairs_hard_v4_holdout}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "?")
case "$GPU_NAME" in
  *"PRO 6000"*|*Blackwell*) OVL="--overlay $BASE/blackwell-torch.overlay:ro"; PY=$VENV/bin/python;;
  *) OVL=""; PY=python3;;
esac
CKPT=$(ls -d "$BASE/runs/$RUN"/checkpoint-g* | sort -V | tail -1)/model.pt
OV=${MODEL_OV:-""}
echo "holdout eval $RUN ckpt=$CKPT gpu=$GPU_NAME pairs=$PAIRS"
# shellcheck disable=SC2086
apptainer exec --nv $OVL "$SIF" "$PY" \
    "$BASE/harness/suite_smoke_logicdiff.py" \
    --ckpt "$CKPT" --config "$BASE/harness/configs/f0-span-v3-role.yaml" \
    --pairs-dir "$PAIRS" \
    --out-dir "$BASE/runs/$RUN/battery_logicdiff_v4holdout" \
    --device cuda ${OV:+--model-override "$OV"} --modes dense
