#!/bin/bash
#SBATCH --job-name=pubmed-mine
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/pubmed_mine_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/pubmed_mine_%j.err

# =============================================================================
# PubMed multi-dominio miner -> eco_corpus.jsonl (corpus dLLM EcoReasoner)
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
MAX="${MAX_PER_DOMAIN:-150000}"
OUT="${CORPUS_OUT:-$BASE/data/eco_corpus.jsonl}"
echo "[$(date)] PubMed mine: max_per_domain=$MAX -> $OUT"
python3 "$BASE/scripts/mine_pubmed_multidomain.py" \
    --max_per_domain "$MAX" --out "$OUT"
echo "[$(date)] DONE mine"