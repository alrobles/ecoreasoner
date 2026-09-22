#!/usr/bin/env bash
#SBATCH --job-name=sft-mdlm
#SBATCH --partition=sixhour
#SBATCH --time=05:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40|a40|pro6000|q8000"
#SBATCH --mem=60G
#SBATCH --signal=B:USR1@120
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/sft_mdlm_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/sft_mdlm_%j.err
# SFT chat v1 sobre MdLMMoE (init g10-ep3-s1 EMA). Receta LLaDA 2.3:
# prompt visible, response enmascarada t~U(0,1), CE solo en response.
# bs=8: los logits fp32 (Bx768x126080) dominan la VRAM — bs16 OOM en 24GB.
# Uso: INIT=... DATA=... OUT=... sbatch --export=ALL sft_mdlm.slurm
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
VENV=/bb/bwvenv
INIT=${INIT:-$BASE/runs/g10-ep3-s1/checkpoint-g23000}
DATA=${DATA:-$BASE/data/sft_ids_v1.npz}
OUT=${OUT:-$BASE/runs/sft-chat-v1}
EPOCHS=${EPOCHS:-3}
EMA_FLAG=${EMA_FLAG:---ema}
BS=${BS:-8}
LR=${LR:-2e-5}
OFFSET=${OFFSET:-0}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "?")
case "$GPU_NAME" in
  *"PRO 6000"*|*Blackwell*) OVL="--overlay $BASE/blackwell-torch.overlay:ro"; PY=$VENV/bin/python;;
  *) OVL=""; PY=python3;;
esac
echo "[sft] gpu=$GPU_NAME init=$INIT data=$DATA out=$OUT epochs=$EPOCHS bs=$BS lr=$LR $EMA_FLAG"
# shellcheck disable=SC2086
apptainer exec --nv $OVL "$SIF" "$PY" $BASE/scripts/sft_mdlm.py \
  --init "$INIT" $EMA_FLAG --data_cache "$DATA" --output "$OUT" \
  --epochs "$EPOCHS" --batch_size "$BS" --lr "$LR" --step_offset "$OFFSET"
echo "=== SFT DONE $(date -u) ==="
