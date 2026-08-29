#!/usr/bin/env bash
#SBATCH --job-name=build-v6
#SBATCH --partition=sixhour
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/build_v6_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/build_v6_%j.err

# =============================================================================
# build_v6.slurm — Construye train_corpus_v6.jsonl = v5 + fulltext PMC 2025-2026.
#  - Dedup por PMCID (v5 gana prioridad, ya etiquetado).
#  - Domain inferido por ILIKE a los docs nuevos sin domain.
#  - Balance opcional por dominio (--max-per-domain).
#  - Escribe train_corpus_v6.jsonl + v6_report.json. NO toca v5 ni el fetch.
# CPU-only (sin GPU). Ejecutar SOLO via Slurm (build_v6.py lo exige).
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"

MAX_PER_DOMAIN="${MAX_PER_DOMAIN:-300000}"
python3 scripts/build_v6.py \
    --v5  "$BASE/data/train_corpus_v5.jsonl" \
    --fetch-dir "$BASE/data/fulltext_2526" \
    --out "$BASE/data/train_corpus_v6.jsonl" \
    --report "$BASE/data/v6_report.json" \
    --max-per-domain "$MAX_PER_DOMAIN" \
    --shuffle --seed 42

echo "build_v6 done rc=$?"