#!/bin/bash
#SBATCH --job-name=pre-tokenize
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=05:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/pre_tokenize_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/pre_tokenize_%j.err

# Pre-tokenizacion del corpus (una sola vez) -> train_ids_v3.npy
# Elimina el re-tokenizado dentro de cada slurm de entrenamiento.
set -u
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
echo "[$(date)] pre-tokenizando train_corpus_v3 -> train_ids_v3.npy"
apptainer exec "$SIF" python3 "$BASE/scripts/pre_tokenize.py" \
    --input "$BASE/data/train_corpus_v3.jsonl" \
    --tokenizer "$TOK" \
    --out "$BASE/data/train_ids_v3.npy" \
    --workers 8 --seq_len 768
echo "[$(date)] PRE-TOKENIZE DONE exit=$?"