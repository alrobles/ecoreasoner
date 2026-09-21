#!/usr/bin/env bash
#SBATCH --job-name=qagen
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40|a40|q8000|pro6000"
#SBATCH --mem=80G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/qagen_%A_%a.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/qagen_%A_%a.err
# QA synth pipeline: teacher OLMo-2-13B-Instruct (Ai2, Apache 2.0)
# genera K pares por pasaje (patrón mt_vllm.slurm). Shard via ARRAY_TASK_ID.
# Constraint >=40GB: 13B bf16 ~26GB no cabe en q6000 (24GB).
# Qwen2.5 quedó vetado por restricción de modelos chinos — output en
# data/qa_raw_QUAR_qwen (no usar para SFT).
set -euo pipefail
BASE=/beegfs/a474r867
SIF=$BASE/ecoreasoner/pytorch-cuda.sif
MODEL=${MODEL:-allenai/OLMo-2-1124-13B-Instruct}
NSHARDS=${NSHARDS:-4}
SHARD=$(( ${SHARD_BASE:-0} + ${SLURM_ARRAY_TASK_ID:-0} ))
INDIR=${INDIR:-$BASE/ecoreasoner/data/qa_passages.jsonl}
OUTDIR=${OUTDIR:-$BASE/ecoreasoner/data/qa_raw}
K=${K:-4}
MAXDOCS=${MAXDOCS:-256}
LIMIT=${LIMIT:-0}
export PYTHONPATH=$BASE/${PYLIBS:-pylibs_vllm}
export HF_HOME=$BASE/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_V1=0  # V1 necesita numba (roto con numpy 2.4); V0 corre en todas las islas
export TRITON_CACHE_DIR=$BASE/triton_cache
echo "[$(hostname)] shard=$SHARD/$NSHARDS model=$MODEL indir=$INDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Turing (q8000=RTX8000 cc7.5) no soporta bf16 -> float16
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
DTYPE=${DTYPE:-bfloat16}
case "$GPU_NAME" in *8000*|*Turing*|*T4*) DTYPE=float16;; esac
echo "dtype=$DTYPE gpu=$GPU_NAME"
apptainer exec --nv "$SIF" python3 "$BASE/ecoreasoner/scripts/qa_gen.py" \
  --indir "$INDIR" --outdir "$OUTDIR" \
  --shard "$SHARD" --nshards "$NSHARDS" \
  --model "$MODEL" --k "$K" --maxdocs "$MAXDOCS" --limit "$LIMIT" \
  --dtype "$DTYPE"
