#!/bin/bash
#SBATCH --job-name=pre-tok-v5
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/pre_tokenize_v5_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/pre_tokenize_v5_%j.err
# Pre-tokenizacion del corpus v5 (train_corpus_v5.jsonl) -> train_ids_v5.npy
# Dataset v5 = v3 + PMC fulltext, dedup global, 12 dominios, 1.9M docs.
# Una sola vez; evita re-tokenizar en cada slurm de entrenamiento.
set -u
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
echo "[$(date)] pre-tokenizando train_corpus_v5 -> train_ids_v5.npy"
apptainer exec "$SIF" python3 "$BASE/scripts/pre_tokenize.py" \
    --input "$BASE/data/train_corpus_v5.jsonl" \
    --tokenizer "$TOK" \
    --out "$BASE/data/train_ids_v5.npy" \
    --workers 16 --seq_len 768
echo "[$(date)] PRE-TOKENIZE v5 DONE exit=$?"