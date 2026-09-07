#!/usr/bin/env bash
#SBATCH --job-name=skel-sample
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/skel_sample_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/skel_sample_%j.err

# F0b: validacion de COBERTURA del extractor de esqueletos sobre el corpus real.
# Procesa hasta N docs por dominio (sample) y reporta cobertura por estrategia.
set -uo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"

python3 "$BASE/scripts/build_skeleton.py" \
    --input "$BASE/data/curate/train_corpus_v7_curated.jsonl" \
    --mode sample \
    --per-domain 300 \
    --min-stages 3 \
    --report "$BASE/data/skeleton/sample_report.json"

echo "=== EXIT $? ==="