#!/usr/bin/env bash
#SBATCH --job-name=dllm-v2-retok-fix
#SBATCH --partition=sixhour
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/dllm_v2_retokenize_fix_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/dllm_v2_retokenize_fix_%j.err
# =============================================================================
# Re-tokeniza train/val de esqueletos con pre_tokenize_v2.py corregido.
# Genera archivos *_fix.npz con nombres distintos al cache vivo.
# CPU-only (no --nv). NO toca train_ids_skeleton_v2.npz en uso.
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
export HF_HUB_OFFLINE=1

mkdir -p "$BASE/data" "$BASE/logs"

RUN="apptainer exec --overlay $OVL:ro $SIF $VENV/bin/python"

echo "=== re-tokenize train $(date -u) ==="
$RUN "$BASE/scripts/pre_tokenize_v2.py" \
    --input "$BASE/data/skeleton/train_skeleton_train.jsonl" \
    --tokenizer "$TOK" \
    --out "$BASE/data/train_ids_skeleton_v2_fix.npz" \
    --seq_len 768 --workers 16

echo "=== re-tokenize val $(date -u) ==="
$RUN "$BASE/scripts/pre_tokenize_v2.py" \
    --input "$BASE/data/skeleton/train_skeleton_val.jsonl" \
    --tokenizer "$TOK" \
    --out "$BASE/data/val_ids_skeleton_v2_fix.npz" \
    --seq_len 768 --workers 4

echo "=== verify $(date -u) ==="
$RUN -c "
import numpy as np
for name in ['train_ids_skeleton_v2_fix.npz', 'val_ids_skeleton_v2_fix.npz']:
    p = '$BASE/data/' + name
    npz = np.load(p)
    vmax = int(npz['ids'].max())
    vocab = int(npz.get('vocab_size', 126080))
    print(f'{name}: max_id={vmax} vocab={vocab} OK={vmax < vocab}')
    assert vmax < vocab, f'{name}: max_id {vmax} >= vocab {vocab}'
"

echo "=== dllm-v2-retok-fix DONE $(date -u) ==="
