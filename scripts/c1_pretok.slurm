#!/usr/bin/env bash
#SBATCH --job-name=c1tok
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/c1tok_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/c1tok_%j.err
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
apptainer exec $BASE/pytorch-cuda.sif python3 $BASE/scripts/pre_tokenize_v2.py   --input $BASE/data/corpus_v3.jsonl --tokenizer $TOK   --out $BASE/data/train_ids_c1.npz --workers 16 --seq_len 768
echo "=== C1TOK DONE ==="
