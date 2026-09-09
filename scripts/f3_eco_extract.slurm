#!/bin/bash
#SBATCH --job-name=f3-eco-extract
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/f3_eco_extract_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/f3_eco_extract_%j.err

# FASE 3: subcorpus ECO del v7 curado (dominios eco/phylo/bioc/climate/conserv/
# marine/plant/palaeo/microbio) -> extractor de entidades sobre el subset real
# ecologico. El PMC completo es biomedico; el subcorpus eco es el objetivo.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
echo "[$(date)] f3-eco-extract start"

python3 "$BASE/scripts/extract_eco_v7.py" \
  --in "$BASE/data/curate/train_corpus_v7_curated.jsonl" \
  --out "$BASE/data/curate/eco_v7_subset.jsonl"

python3 "$BASE/scripts/extract_entities_lit.py" \
  --input "$BASE/data/curate/eco_v7_subset.jsonl" \
  --sample 400000 --top 200000 \
  --out "$BASE/docs/results/entities_eco_v7.json"

echo "[$(date)] f3-eco-extract DONE rc=$?"