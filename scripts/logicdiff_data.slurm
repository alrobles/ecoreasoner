#!/bin/bash
#SBATCH --job-name=logicdiff-data
#SBATCH --partition=sixhour
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/logicdiff_data_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/logicdiff_data_%j.err

# Dataset de roles logicos (LogicDiff Fase A) sobre skeleton train.
# CPU-only: batch-encoding fast tokenizer + reglas de etapa/conectivas.

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07

export TOKENIZERS_PARALLELISM=false

echo "=== logicdiff dataset build $(date -u) ==="

apptainer exec --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python -u \
    "$BASE/scripts/build_logicdiff_dataset.py" \
    --in "$BASE/data/skeleton/train_skeleton_train.jsonl" \
    --tokenizer "$TOK" \
    --out "$BASE/data/logicdiff/role_train.pt" \
    --max-len 768 ${LIMIT:+--limit $LIMIT}

echo "=== DONE $(date -u) ==="
