#!/usr/bin/env bash
#SBATCH --job-name=eval-generic
#SBATCH --partition=sixhour
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/eval_generic_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/eval_generic_%j.err
# Eval logicdiff generica: cualquier RUNDIR/checkpoint-g* sobre cualquier
# PAIRS de pares, con cualquier CFG de eval. Variante parametrizable de
# g_eval_holdout_v4.slurm (que tenia out-dir y pairs fijos).
# Uso: RUNDIR=... PAIRS=... CFG=... OUTDIR_NAME=... [MODES=dense,base]
#      sbatch --export=ALL g_eval_generic.slurm
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
VENV=/bb/bwvenv
RUNDIR=${RUNDIR:?falta RUNDIR}
PAIRS=${PAIRS:-$BASE/runs/pairs_hard_v4_holdout_clean}
CFG=${CFG:-$BASE/harness/configs/eval-moe-v4.yaml}
OUTDIR_NAME=${OUTDIR_NAME:-battery_logicdiff_generic}
MODES=${MODES:-dense}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "?")
case "$GPU_NAME" in
  *"PRO 6000"*|*Blackwell*) OVL="--overlay $BASE/blackwell-torch.overlay:ro"; PY=$VENV/bin/python;;
  *) OVL=""; PY=python3;;
esac
CKPT=$(ls -d "$RUNDIR"/checkpoint-g* | sort -V | tail -1)/model.pt
echo "eval $RUNDIR ckpt=$CKPT gpu=$GPU_NAME pairs=$PAIRS modes=$MODES -> $OUTDIR_NAME"
# shellcheck disable=SC2086
apptainer exec --nv $OVL "$SIF" "$PY" \
    "$BASE/harness/suite_smoke_logicdiff.py" \
    --ckpt "$CKPT" --config "$CFG" \
    --pairs-dir "$PAIRS" \
    --out-dir "$RUNDIR/$OUTDIR_NAME" \
    --device cuda --modes "$MODES"
