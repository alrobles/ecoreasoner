#!/usr/bin/env bash
#SBATCH --job-name=holdout-v5
#SBATCH --partition=sixhour
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/holdout_v5_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/holdout_v5_%j.err
# Eval holdout_clean del backbone v5-1b / sft-v1 (MoE 768/12L/12H/16E top-1).
# Variante de g_eval_holdout_v4.slurm con config eval-moe-v5.yaml (16E) y
# soporte EMA (default ON — los evals corren sobre EMA).
# Uso: RUNDIR=... [EMA=0] sbatch --export=ALL g_eval_holdout_v5.slurm
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
VENV=/bb/bwvenv
RUNDIR=${RUNDIR:-$BASE/runs/sft-v1}
PAIRS=${PAIRS:-$BASE/runs/pairs_hard_v4_holdout_clean}
EMA=${EMA:-1}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "?")
case "$GPU_NAME" in
  *"PRO 6000"*|*Blackwell*) OVL="--overlay $BASE/blackwell-torch.overlay:ro"; PY=$VENV/bin/python;;
  *) OVL=""; PY=python3;;
esac
CKPT_DIR=$(ls -d "$RUNDIR"/checkpoint-g* | sort -V | tail -1)
if [ "$EMA" = "1" ] && [ -f "$CKPT_DIR/ema_model.pt" ]; then
    CKPT=$CKPT_DIR/ema_model.pt
else
    CKPT=$CKPT_DIR/model.pt
fi
echo "holdout eval $RUNDIR ckpt=$CKPT gpu=$GPU_NAME pairs=$PAIRS"
# shellcheck disable=SC2086
apptainer exec --nv $OVL "$SIF" "$PY" \
    "$BASE/harness/suite_smoke_logicdiff.py" \
    --ckpt "$CKPT" --config "$BASE/harness/configs/eval-moe-v5.yaml" \
    --pairs-dir "$PAIRS" \
    --out-dir "$RUNDIR/battery_logicdiff_v4holdout" \
    --device cuda --modes dense
