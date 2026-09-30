#!/usr/bin/env bash
#SBATCH --job-name=sft-v2
#SBATCH --partition=sixhour
#SBATCH --time=05:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40|a40|pro6000|q8000"
#SBATCH --mem=60G
#SBATCH --signal=B:USR1@120
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/sft_v2_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/sft_v2_%j.err
# SFT chat v2 sobre backbone v5-1b MADURO (gate: ckpt >=g700K o post-1.3M).
# Receta LLaDA 2.3 (igual que sft-v1): prompt visible, response enmascarada
# t~U(0,1), CE solo en response. Trainer infiere arch del ckpt (16E v5 OK).
# sft_mdlm.py no auto-resume: las olas las encadena watch_sft_v2.sh con
# INIT=ultimo ckpt + OFFSET=ultimo gN (numeracion acumulativa).
# Uso primera ola:  INIT=<ckpt dir con ema_model.pt> sbatch sft_mdlm_v5.slurm
# Uso smoke 15min:  INIT=... OUT=runs/sft-v2-smoke sbatch --time=00:15:00
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
VENV=/bb/bwvenv
INIT=${INIT:?'INIT requerido: dir checkpoint-gN con ema_model.pt del pretrain v5'}
DATA=${DATA:-$BASE/data/sft_ids_v1.npz}
OUT=${OUT:-$BASE/runs/sft-v2}
EPOCHS=${EPOCHS:-3}
EMA_FLAG=${EMA_FLAG:---ema}
BS=${BS:-8}
LR=${LR:-2e-5}
OFFSET=${OFFSET:-0}
SAVE_EVERY=${SAVE_EVERY:-500}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "?")
case "$GPU_NAME" in
  *"PRO 6000"*|*Blackwell*) OVL="--overlay $BASE/blackwell-torch.overlay:ro"; PY=$VENV/bin/python;;
  *) OVL=""; PY=python3;;
esac
echo "[sft-v2] gpu=$GPU_NAME init=$INIT data=$DATA out=$OUT epochs=$EPOCHS bs=$BS lr=$LR offset=$OFFSET $EMA_FLAG"
# shellcheck disable=SC2086
apptainer exec --nv $OVL "$SIF" "$PY" $BASE/scripts/sft_mdlm.py \
  --init "$INIT" $EMA_FLAG --data_cache "$DATA" --output "$OUT" \
  --epochs "$EPOCHS" --batch_size "$BS" --lr "$LR" --step_offset "$OFFSET" \
  --save_every "$SAVE_EVERY"
