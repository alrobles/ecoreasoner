#!/usr/bin/env bash
#SBATCH --job-name=skel-full
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/skel_full_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/skel_full_%j.err

# F0b fase 2: build FULL del corpus de esqueletos a partir del v7_curated.
# Filtra docs con >=3 etapas, balancea PER_DOMAIN max por dominio, escribe
# data/skeleton/train_skeleton.jsonl + reporte. 0 GPU, CPU only.
set -uo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"

python3 "$BASE/scripts/build_skeleton.py" \
    --input "$BASE/data/curate/train_corpus_v7_curated.jsonl" \
    --mode full \
    --out "$BASE/data/skeleton/train_skeleton.jsonl" \
    --per-domain 50000 \
    --min-stages 3 \
    --report "$BASE/data/skeleton/full_report.json"

echo "=== EXIT $? ==="