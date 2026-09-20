#!/usr/bin/env bash
# DEPRECATED 2026-09-20: dependia de un modelo de origen chino vetado
# (Qwen / Tencent HY-MT / LLaDA weights / DeepSeek). Los pesos fueron
# eliminados del cluster; este script ya no corre. El tokenizer LLaDA
# (vocab 126080) sigue vivo SOLO como tokenizer de nuestro modelo.
#SBATCH --job-name=mtvllm
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/mtvllm_%A_%a.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/mtvllm_%A_%a.err
# vLLM offline batch ES->EN con HY-MT1.5. Override: --gres=gpu:pro6000:1
set -euo pipefail
BASE=/beegfs/a474r867
SIF=$BASE/ecoreasoner/pytorch-cuda.sif
MODEL=${MODEL:-tencent/HY-MT1.5-7B}
NSHARDS=${NSHARDS:-6}
LIMIT=${LIMIT:-0}
CHUNK=${CHUNK:-8000}
MAXDOCS=${MAXDOCS:-64}
MAXLEN=${MAXLEN:-8192}
GMU=${GMU:-0.92}
INDIR=${INDIR:-$BASE/unam_mt_in}
OUTDIR=${OUTDIR:-$BASE/unam_md_en}
REPORT=${REPORT:-$OUTDIR/mt_report.jsonl}

export PYTHONPATH=$BASE/${PYLIBS:-pylibs_vllm}
export HF_HOME=$BASE/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING
export CC=$BASE/conda_cc/bin/x86_64-conda-linux-gnu-cc
export CXX=$BASE/conda_cc/bin/x86_64-conda-linux-gnu-c++
export TRITON_CACHE_DIR=$BASE/triton_cache

mkdir -p "$OUTDIR" "$BASE/ecoreasoner/logs"
SHARD=$(( ${SHARD_BASE:-0} + SLURM_ARRAY_TASK_ID ))
echo "[$(hostname)] shard=$(( ${SHARD_BASE:-0} + SLURM_ARRAY_TASK_ID ))/$NSHARDS model=$MODEL limit=$LIMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv "$SIF" python3 "$BASE/ecoreasoner/scripts/traducir_vllm.py" \
  --indir "$INDIR" --outdir "$OUTDIR" \
  --shard "$SHARD" --nshards "$NSHARDS" \
  --model "$MODEL" --chunk "$CHUNK" --maxdocs "$MAXDOCS" \
  --maxlen "$MAXLEN" --gmu "$GMU" --limit "$LIMIT" \
  --report "$REPORT"
