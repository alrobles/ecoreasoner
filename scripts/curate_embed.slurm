#!/usr/bin/env bash
#SBATCH --job-name=curate-emb
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=48G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/curate_emb_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/curate_emb_%j.err

# FASE 1: embeddings BGE-small en GPU de los 2.2M docs -> vectors_full.npy
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
export HF_HOME=/beegfs/a474r867/hf-cache
export CUDA_VISIBLE_DEVICES=0
"$BASE/venv_curate/bin/python" "$BASE/scripts/curate_final.py" \
  --v6 "$BASE/data/train_corpus_v6.jsonl" \
  --phys "$BASE/data/train_corpus_phys.jsonl" \
  --vecs "$BASE/data/curate/vectors_full.npy" \
  --batch 1024 --only-embed
echo "curate-emb done rc=$?"