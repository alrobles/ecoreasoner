#!/bin/bash
#SBATCH --job-name=f3-eco-fine
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/f3_eco_fine_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/f3_eco_fine_%j.err

# FASE 3: subcorpus ECO GENUINO del v5 por domain_fine (MeSH+texto, ~40 dominios).
# Los dominios gruesos del v7 NO separan eco de biomed (hallazgo 2026-09-09);
# la etiqueta FINA del v5 sí. Extrae los 17 dominios finos ecológicos de campo.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
echo "[$(date)] f3-eco-fine start"

python3 "$BASE/scripts/extract_eco_v5_fine.py" \
  --in "$BASE/data/train_corpus_v5.jsonl" \
  --out "$BASE/data/curate/eco_v5_fine.jsonl" | tee "$BASE/logs/f3_eco_fine_$SLURM_JOB_ID.filter.log"

echo "[$(date)] f3-eco-fine DONE rc=$?"