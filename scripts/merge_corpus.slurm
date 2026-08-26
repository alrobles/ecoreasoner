#!/bin/bash
#SBATCH --job-name=merge-corpus
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/merge_corpus_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/merge_corpus_%j.err

# =============================================================================
# Merge final del nUEVO corpus ampliado:
#   fulltext_corpus_all.jsonl (97,012 PMC) + eco_corpus_v2.jsonl (1M abstracts)
#   -> train_corpus_v2.jsonl. Full-text prioridad; abstracts sin fulltext.
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
echo "[$(date)] merge corpus ampliado"
python3 "$BASE/scripts/merge_corpus.py" \
  --abstracts "$BASE/data/eco_corpus_v2.jsonl" \
  --fulltext "$BASE/data/fulltext_corpus_all.jsonl" \
  --out "$BASE/data/train_corpus_v2.jsonl" --shuffle --seed 42
echo "[$(date)] DONE"