#!/bin/bash
#SBATCH --job-name=f3-extract-ents
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/f3_extract_ents_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/f3_extract_ents_%j.err

# FASE 3: extraccion de entidades (especies/regiones/datasets) del fulltext real.
# CPU-only, 1 job, muestra = corpus completo (97K docs).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
echo "[$(date)] f3-extract-ents start"
python3 "$BASE/scripts/extract_entities_lit.py" \
  --input "$BASE/data/fulltext_corpus_all.jsonl" \
  --sample 97012 --top 200000 \
  --out "/beegfs/a474r867/ecoreasoner/docs/results/entities_fulltext_corpus.json"
echo "[$(date)] f3-extract-ents DONE rc=$?"