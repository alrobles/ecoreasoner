#!/usr/bin/env bash
#SBATCH --job-name=c1emb
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/c1emb_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/c1emb_%j.err
# c1: embeddings de skeleton_v2 + holdout + UNAM, luego audit
# (dedup + leakage + kNN graph). transformers 4.46 via pylibs -> corre
# en cualquier GPU del cluster (no depende de pro6000).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
VENV=/bb/bwvenv
PYLIBS=/beegfs/a474r867/pylibs
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
EMB=$BASE/emb_v1
mkdir -p "$EMB" "$BASE/logs"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "?")
case "$GPU_NAME" in
  *"PRO 6000"*|*Blackwell*) OVL_ARGS="--overlay $BASE/blackwell-torch.overlay:ro"; PYBIN="$VENV/bin/python";;
  *) OVL_ARGS=""; PYBIN=python3;;
esac
echo "GPU: $GPU_NAME PY=$PYBIN"
export PYTHONPATH=$PYLIBS
export HF_HOME=/beegfs/a474r867/hf-cache
export TOKENIZERS_PARALLELISM=false

PY="apptainer exec --nv $OVL_ARGS $SIF $PYBIN"

# --- 0. prep: decodifica holdout a texto + unam md a chunks jsonl ---
$PY $BASE/scripts/prep_embed_inputs.py \
  --holdout $BASE/runs/pairs_hard_v4_holdout --tokenizer $TOK \
  --unam-md '/beegfs/a474r867/curada_v2/science/*.md' \
  --unam-meta /beegfs/a474r867/curada_v2/meta.jsonl \
  --out $EMB/inputs

# --- 1. embed skeleton_v2 (corpus activo) ---
$PY $BASE/scripts/embed_docs.py \
  --jsonl $BASE/data/skeleton/train_skeleton_train.jsonl \
  --out-dir $EMB/skeleton_v2 --batch 384

# --- 2. embed holdout ---
if [ -f $EMB/inputs/holdout_docs.jsonl ]; then
  $PY $BASE/scripts/embed_docs.py \
    --jsonl $EMB/inputs/holdout_docs.jsonl --text-field text \
    --meta-fields "level,src" --out-dir $EMB/holdout
fi

# --- 3. embed unam ---
if [ -f $EMB/inputs/unam_docs.jsonl ]; then
  $PY $BASE/scripts/embed_docs.py \
    --jsonl $EMB/inputs/unam_docs.jsonl --text-field text \
    --meta-fields "pid,domain,lang,src,facultad" --out-dir $EMB/unam
fi

# --- 4. audit: dedup + leakage + kNN ---
if [ -f $EMB/holdout/emb.fp16.npy ]; then HO="--holdout-emb-dir $EMB/holdout"; else HO=""; fi
$PY $BASE/scripts/corpus_audit.py \
  --emb-dir $EMB/skeleton_v2 $HO --out $EMB/audit_skeleton_v2 --knn-k 16
if [ -f $EMB/unam/emb.fp16.npy ]; then
  $PY $BASE/scripts/corpus_audit.py \
    --emb-dir $EMB/unam $HO --out $EMB/audit_unam --knn-k 8
fi
echo "=== C1EMB DONE $(date -u) ==="
