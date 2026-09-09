#!/bin/bash
#SBATCH --job-name=f3-eco-fine-extract
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/f3_eco_fine_extract_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/f3_eco_fine_extract_%j.err

# FASE 3: extractor de entidades sobre el subcorpus ECO GENUINO (domain_fine v5).
# Se lanza con --dependency=afterok:<filtrado> ; si el filtrado falla, no corre.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
echo "[$(date)] f3-eco-fine-extract start"

python3 "$BASE/scripts/extract_entities_lit.py" \
  --input "$BASE/data/curate/eco_v5_fine.jsonl" \
  --sample 400000 --top 200000 \
  --out "$BASE/docs/results/entities_eco_fine_v5.json"

echo "[$(date)] f3-eco-fine-extract DONE rc=$?"