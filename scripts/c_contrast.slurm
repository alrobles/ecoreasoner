#!/bin/bash
#SBATCH --job-name=c-eval
#SBATCH --output=logs/c-eval-%j.out
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
# Eval de contraste C1-C4: ckpt propio o backend HF externo (LLaDA-8B).
# Uso:
#   RUNDIR=runs/b2-qa PAIRS=runs/pairs_l3_fluent MODES=consistency_profile sbatch c_contrast.slurm
#   HF_MODEL=/path/llada PAIRS=runs/pairs_l3_phys MODES=consistency sbatch c_contrast.slurm
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
VENV=/bb/bwvenv
PAIRS=${PAIRS:-$BASE/runs/pairs_l3_fluent}
MODES=${MODES:-consistency}
EMA=${EMA:-1}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "?")
case "$GPU_NAME" in
  *"PRO 6000"*|*Blackwell*) OVL="--overlay $BASE/blackwell-torch.overlay:ro"; PY=$VENV/bin/python;;
  *) OVL=""; PY=python3;;
esac
EXTRA=""
if [ -n "${HF_MODEL:-}" ]; then
    CKPT=""
    EXTRA="--hf_model $HF_MODEL"
    TAG="hf-$(basename "$HF_MODEL" | cut -c1-12)"
    # transformers 4.x compatible con torch 2.4 del sif (la de ~/.local exige >=2.5)
    export PYTHONPATH=/beegfs/a474r867/pydeps_tf451${PYTHONPATH:+:$PYTHONPATH}
else
    RUNDIR=${RUNDIR:-$BASE/runs/v5-1b}
    CKPT_DIR=${CKPT_DIR:-$(ls -d "$RUNDIR"/checkpoint-g* | sort -V | tail -1)}
    if [ "$EMA" = "1" ] && [ -f "$CKPT_DIR/ema_model.pt" ]; then
        CKPT=$CKPT_DIR/ema_model.pt
    else
        CKPT=$CKPT_DIR/model.pt
    fi
    TAG="$(basename "$RUNDIR")-$(basename "$CKPT_DIR")"
fi
OUTDIR=${OUTDIR:-$BASE/runs/contrast/$(basename "$PAIRS")_${TAG}}
echo "contrast eval tag=$TAG ckpt=${CKPT:-none} hf=${HF_MODEL:-none} gpu=$GPU_NAME pairs=$PAIRS modes=$MODES out=$OUTDIR"
ARGS=(--config "$BASE/harness/configs/eval-moe-v5.yaml" --pairs-dir "$PAIRS"
      --out-dir "$OUTDIR" --device cuda --modes "$MODES")
[ -n "$CKPT" ] && ARGS+=(--ckpt "$CKPT")
# shellcheck disable=SC2086
apptainer exec --nv $OVL "$SIF" "$PY" \
    "$BASE/harness/suite_smoke_logicdiff.py" "${ARGS[@]}" $EXTRA
