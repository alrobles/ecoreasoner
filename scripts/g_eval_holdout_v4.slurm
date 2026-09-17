#!/usr/bin/env bash
#SBATCH --job-name=holdout-v4
#SBATCH --partition=sixhour
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/holdout_v4_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/holdout_v4_%j.err
# Eval holdout del prototipo v4 (MoE 768/12L/12H/8E) sobre el holdout limpio.
# Variante de g_eval_holdout.slurm: config propia eval-moe-v4.yaml (sin
# MODEL_OV — el JSON se manglaba por ssh/sbatch) y OUT en outputs/.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
VENV=/bb/bwvenv
RUN=${RUN:-retrain_v4}
RUNDIR=${RUNDIR:-$BASE/outputs/$RUN}
PAIRS=${PAIRS:-$BASE/runs/pairs_hard_v4_holdout_clean}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "?")
case "$GPU_NAME" in
  *"PRO 6000"*|*Blackwell*) OVL="--overlay $BASE/blackwell-torch.overlay:ro"; PY=$VENV/bin/python;;
  *) OVL=""; PY=python3;;
esac
CKPT=$(ls -d "$RUNDIR"/checkpoint-g* | sort -V | tail -1)/model.pt
echo "holdout eval $RUN ckpt=$CKPT gpu=$GPU_NAME pairs=$PAIRS"
# shellcheck disable=SC2086
apptainer exec --nv $OVL "$SIF" "$PY" \
    "$BASE/harness/suite_smoke_logicdiff.py" \
    --ckpt "$CKPT" --config "$BASE/harness/configs/eval-moe-v4.yaml" \
    --pairs-dir "$PAIRS" \
    --out-dir "$RUNDIR/battery_logicdiff_v4holdout" \
    --device cuda --modes dense
