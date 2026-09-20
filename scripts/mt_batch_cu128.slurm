#!/usr/bin/env bash
# DEPRECATED 2026-09-20: dependia de un modelo de origen chino vetado
# (Qwen / Tencent HY-MT / LLaDA weights / DeepSeek). Los pesos fueron
# eliminados del cluster; este script ya no corre. El tokenizer LLaDA
# (vocab 126080) sigue vivo SOLO como tokenizer de nuestro modelo.
#SBATCH --job-name=mt7bc
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/mt7bc_%A_%a.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/mt7bc_%A_%a.err
# mt_batch con torch 2.7.0+cu128 (pylibs_cu128) dentro del SIF —
# soporta a100 y pro6000(Blackwell sm_120). Override: --gres=gpu:a100:1
set -euo pipefail
BASE=/beegfs/a474r867
SIF=$BASE/ecoreasoner/pytorch-cuda.sif
MODEL=${MODEL:-tencent/HY-MT1.5-7B}
NSHARDS=${NSHARDS:-6}
LIMIT=${LIMIT:-0}
BATCH=${BATCH:-8}
CHUNK=${CHUNK:-8000}
DTYPE=${DTYPE:-bf16}
TOKBUDGET=${TOKBUDGET:-8000}
INDIR=${INDIR:-$BASE/unam_mt_in}
OUTDIR=${OUTDIR:-$BASE/unam_md_en}
REPORT=${REPORT:-$OUTDIR/mt_report.jsonl}

export PYTHONPATH=$BASE/pylibs_cu128
export HF_HOME=$BASE/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTDIR" "$BASE/ecoreasoner/logs"
SHARD=$(( ${SHARD_BASE:-0} + SLURM_ARRAY_TASK_ID ))
echo "[$(hostname)] shard=$(( ${SHARD_BASE:-0} + SLURM_ARRAY_TASK_ID ))/$NSHARDS model=$MODEL limit=$LIMIT dtype=$DTYPE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv "$SIF" python3 "$BASE/ecoreasoner/scripts/traducir_batch.py" \
  --indir "$INDIR" --outdir "$OUTDIR" \
  --shard "$SHARD" --nshards "$NSHARDS" \
  --model "$MODEL" --batch "$BATCH" --chunk "$CHUNK" --dtype "$DTYPE" \
  --tokbudget "$TOKBUDGET" --limit "$LIMIT" ${NOPENALTY:+--nopenalty} \
  --report "$REPORT"
