#!/usr/bin/env bash
#SBATCH --job-name=build-v5
#SBATCH --partition=sixhour
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/build_v5_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/build_v5_%j.err

# =============================================================================
# build_v5.slurm — Construye train_corpus_v5.jsonl (Opción A: corpus limpio)
#  - Dedup global por pmid con prioridad full-text (PMC-v4 gana a v3).
#  - Asignacion de domain por ILIKE a los docs sin domain (PMC).
#  - Balance opcional por dominio (--max-per-domain).
#  - Escribe train_corpus_v5.jsonl + v5_report.json. NO toca v4 ni el training.
# CPU-only (sin GPU). Ejecutar SOLO via Slurm (build_v5.py lo exige).
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"

MAX_PER_DOMAIN="${MAX_PER_DOMAIN:-300000}"
python3 scripts/build_v5.py \
    --v3  "$BASE/data/train_corpus_v3.jsonl" \
    --pmc "$BASE/data/pmc_corpus_v4" \
    --out "$BASE/data/train_corpus_v5.jsonl" \
    --report "$BASE/data/v5_report.json" \
    --max-per-domain "$MAX_PER_DOMAIN" \
    --shuffle --seed 42

echo "build_v5 done rc=$?"