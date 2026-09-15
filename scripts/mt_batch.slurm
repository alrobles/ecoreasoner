#!/usr/bin/env bash
#SBATCH --job-name=mt7b
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/mt7b_%A_%a.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/mt7b_%A_%a.err
# Traducción batch ES->EN con HY-MT1.5-7B. Array: cada task = shard.
# sbatch --array=0-(N-1) mt_batch.slurm   |   NSHARDS, LIMIT por env.
set -euo pipefail
BASE=/beegfs/a474r867
SIF=$BASE/ecoreasoner/pytorch-cuda.sif
MODEL=${MODEL:-tencent/HY-MT1.5-7B}
NSHARDS=${NSHARDS:-6}
LIMIT=${LIMIT:-0}
BATCH=${BATCH:-8}
CHUNK=${CHUNK:-8000}
DTYPE=${DTYPE:-bf16}
TOKBUDGET=${TOKBUDGET:-6000}
INDIR=${INDIR:-$BASE/unam_mt_in}
OUTDIR=${OUTDIR:-$BASE/unam_md_en}
REPORT=${REPORT:-$OUTDIR/mt_report.jsonl}

export PYTHONPATH=$BASE/pylibs_mt
export HF_HOME=$BASE/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUTDIR" "$BASE/ecoreasoner/logs"
SHARD=$(( ${SHARD_BASE:-0} + SLURM_ARRAY_TASK_ID ))
echo "[$(hostname)] shard=$(( ${SHARD_BASE:-0} + SLURM_ARRAY_TASK_ID ))/$NSHARDS model=$MODEL limit=$LIMIT"

apptainer exec --nv "$SIF" python3 "$BASE/ecoreasoner/scripts/traducir_batch.py" \
  --indir "$INDIR" --outdir "$OUTDIR" \
  --shard "$SHARD" --nshards "$NSHARDS" \
  --model "$MODEL" --batch "$BATCH" --chunk "$CHUNK" --dtype "$DTYPE" \
  --tokbudget "$TOKBUDGET" --limit "$LIMIT" \
  --report "$REPORT"
