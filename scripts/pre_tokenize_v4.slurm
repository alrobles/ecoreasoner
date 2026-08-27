#!/bin/bash
#SBATCH --job-name=pre-tok-v4
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/pre_tokenize_v4_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/pre_tokenize_v4_%j.err

# Pre-tokenizacion del corpus v4 (train_corpus_v4.jsonl) -> train_ids_v4.npy
# Dataset v4 = v3 (1.01M docs) + PMC fulltext (1.74M docs) ~ 4-5B tokens.
# Una sola vez; evita re-tokenizar en cada slurm de entrenamiento.
set -u
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
echo "[$(date)] pre-tokenizando train_corpus_v4 -> train_ids_v4.npy"
apptainer exec "$SIF" python3 "$BASE/scripts/pre_tokenize.py" \
    --input "$BASE/data/train_corpus_v4.jsonl" \
    --tokenizer "$TOK" \
    --out "$BASE/data/train_ids_v4.npy" \
    --workers 16 --seq_len 768
echo "[$(date)] PRE-TOKENIZE v4 DONE exit=$?"