#!/usr/bin/env bash
#SBATCH --job-name=evalqa-sft
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40|a40|pro6000|q8000|q6000"
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/evalqa_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/evalqa_%j.err
# Eval generativa de sft-v1 sobre eval_devin_hard (3000 q held-out).
# Resume por pid: si se corta la ventana, relanzar el mismo comando
# continua donde quedo (salta pids ya en --out).
# Uso: CKPT=... sbatch --export=ALL eval_qa_sft.slurm
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
VENV=/bb/bwvenv
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
CKPT=${CKPT:-$BASE/runs/sft-v1}
EVAL=${EVAL:-$BASE/data/eval_devin_hard.jsonl}
OUT=${OUT:-$CKPT/eval_devin_hard.gen.jsonl}
STEPS=${STEPS:-24}
MAXNEW=${MAXNEW:-128}
TEMP=${TEMP:-0.7}
EMA_FLAG=${EMA_FLAG:---ema}
EXTRA=${EXTRA:-}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "?")
case "$GPU_NAME" in
  *"PRO 6000"*|*Blackwell*) OVL="--overlay $BASE/blackwell-torch.overlay:ro"; PY=$VENV/bin/python;;
  *) OVL=""; PY=python3;;
esac
echo "[evalqa] gpu=$GPU_NAME ckpt=$CKPT eval=$EVAL out=$OUT steps=$STEPS maxnew=$MAXNEW temp=$TEMP $EMA_FLAG $EXTRA"
# shellcheck disable=SC2086
apptainer exec --nv $OVL "$SIF" "$PY" $BASE/scripts/eval_qa_sft.py \
  --ckpt "$CKPT" $EMA_FLAG --tokenizer "$TOK" --eval "$EVAL" \
  --out "$OUT" --summary "$OUT.summary.json" \
  --steps "$STEPS" --max-new "$MAXNEW" --temp "$TEMP" $EXTRA
echo "=== EVALQA DONE $(date -u) ==="
