#!/usr/bin/env bash
#SBATCH --job-name=c1audit
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/c1audit_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/c1audit_%j.err
# c1 audit-only: dedup + leakage + kNN sobre embeddings ya calculados
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
VENV=/bb/bwvenv
EMB=$BASE/emb_v1
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "?")
case "$GPU_NAME" in
  *"PRO 6000"*|*Blackwell*) OVL_ARGS="--overlay $BASE/blackwell-torch.overlay:ro"; PYBIN="$VENV/bin/python";;
  *) OVL_ARGS=""; PYBIN=python3;;
esac
echo "GPU: $GPU_NAME PY=$PYBIN"
export PYTHONPATH=/beegfs/a474r867/pylibs
PY="apptainer exec --nv $OVL_ARGS $SIF $PYBIN"
HO=""; [ -f $EMB/holdout/emb.fp16.npy ] && HO="--holdout-emb-dir $EMB/holdout"
$PY $BASE/scripts/corpus_audit.py --emb-dir $EMB/skeleton_v2 $HO \
  --out $EMB/audit_skeleton_v2 --knn-k 16
$PY $BASE/scripts/corpus_audit.py --emb-dir $EMB/unam $HO \
  --out $EMB/audit_unam --knn-k 8
echo "=== C1AUDIT DONE $(date -u) ==="
