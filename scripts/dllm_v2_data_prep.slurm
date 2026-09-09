#!/usr/bin/env bash
#SBATCH --job-name=dllm-v2-prep
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/dllm_v2_prep_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/dllm_v2_prep_%j.err
# =============================================================================
# Preparacion datos + bateria L0-L3 para piloto dLLM v2 (Fase 3 dLLM).
# CPU-only (no --nv). Pasos:
#   1. split train/val del esqueleto;
#   2. pre-tokenizar train y val a .npz (packing EOS);
#   3. generar pares hard L0-L3 desde val.
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
export HF_HUB_OFFLINE=1

mkdir -p "$BASE/data/skeleton" "$BASE/data" "$BASE/runs/pairs_hard_v2" "$BASE/logs"

RUN="apptainer exec --overlay $OVL:ro $SIF $VENV/bin/python"

echo "=== split train/val $(date -u) ==="
$RUN "$BASE/scripts/split_jsonl.py" \
    --input "$BASE/data/skeleton/train_skeleton.jsonl" \
    --train-out "$BASE/data/skeleton/train_skeleton_train.jsonl" \
    --val-out "$BASE/data/skeleton/train_skeleton_val.jsonl" \
    --train-ratio 0.95 --seed 7331

echo "=== pre-tokenize train $(date -u) ==="
$RUN "$BASE/scripts/pre_tokenize_v2.py" \
    --input "$BASE/data/skeleton/train_skeleton_train.jsonl" \
    --tokenizer "$TOK" \
    --out "$BASE/data/train_ids_skeleton_v2.npz" \
    --seq_len 768 --workers 16

echo "=== pre-tokenize val $(date -u) ==="
$RUN "$BASE/scripts/pre_tokenize_v2.py" \
    --input "$BASE/data/skeleton/train_skeleton_val.jsonl" \
    --tokenizer "$TOK" \
    --out "$BASE/data/val_ids_skeleton_v2.npz" \
    --seq_len 768 --workers 4

echo "=== build L0-L3 hard pairs $(date -u) ==="
$RUN "$BASE/harness/build_pairs_hard.py" \
    --in "$BASE/data/skeleton/train_skeleton_val.jsonl" \
    --tokenizer "$TOK" \
    --n 512 --max-len 768 \
    --out "$BASE/runs/pairs_hard_v2" --seed 7331

echo "=== dllm-v2-prep DONE $(date -u) ==="
