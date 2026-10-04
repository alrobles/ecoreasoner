#!/usr/bin/env bash
#SBATCH --job-name=mtolmo
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40|a40|pro6000"
#SBATCH --mem=80G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/mtolmo_%A_%a.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/mtolmo_%A_%a.err
# Traduccion ES->EN de tesis UNAM con teacher COMPLIANT:
# OLMo-2-1124-13B-Instruct (Ai2, Apache 2.0) via traducir_vllm.py.
# Sucede a mt_vllm.slurm (HY-MT purgado 09-20). Stack = el de qa_gen
# (pylibs_vllm + VLLM_USE_V1=0 + HF offline). Constraint >=40GB bf16
# (13B bf16 ~26GB; q6000/q8000 excluidas: Turing sin bf16).
# Resume por .en.md -> seguro re-lanzar el array tras cada ventana.
# Uso:  sbatch --array=0-7 mt_olmo.slurm        (NSHARDS default 8)
#   INDIR/OUTDIR/NSHARDS/LIMIT/MAXDOCS por --export=ALL,...
set -euo pipefail
BASE=/beegfs/a474r867
SIF=$BASE/ecoreasoner/pytorch-cuda.sif
MODEL=${MODEL:-allenai/OLMo-2-1124-13B-Instruct}
NSHARDS=${NSHARDS:-8}
LIMIT=${LIMIT:-0}
CHUNK=${CHUNK:-8000}
MAXDOCS=${MAXDOCS:-64}
MAXLEN=${MAXLEN:-8192}
GMU=${GMU:-0.92}
INDIR=${INDIR:-$BASE/unam_mt_in2}
OUTDIR=${OUTDIR:-$BASE/unam_md_en}
REPORT=${REPORT:-$OUTDIR/mt_olmo_report.jsonl}

export PYTHONPATH=$BASE/${PYLIBS:-pylibs_vllm}
export HF_HOME=$BASE/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_V1=0   # V1 necesita numba (roto con numpy 2.4); V0 en todas
export TRITON_CACHE_DIR=$BASE/triton_cache

mkdir -p "$OUTDIR" "$BASE/ecoreasoner/logs"
SHARD=$(( ${SHARD_BASE:-0} + ${SLURM_ARRAY_TASK_ID:-0} ))
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
DTYPE=${DTYPE:-bfloat16}
case "$GPU_NAME" in *8000*|*Turing*|*T4*) DTYPE=float16;; esac
echo "[$(hostname)] shard=$SHARD/$NSHARDS model=$MODEL dtype=$DTYPE gpu=$GPU_NAME indir=$INDIR"

apptainer exec --nv "$SIF" python3 "$BASE/ecoreasoner/scripts/traducir_vllm.py" \
  --indir "$INDIR" --outdir "$OUTDIR" \
  --shard "$SHARD" --nshards "$NSHARDS" \
  --model "$MODEL" --dtype "$DTYPE" --chunk "$CHUNK" --maxdocs "$MAXDOCS" \
  --maxlen "$MAXLEN" --gmu "$GMU" --limit "$LIMIT" \
  --report "$REPORT"
