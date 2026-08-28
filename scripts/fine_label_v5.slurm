#!/usr/bin/env bash
#SBATCH --job-name=fine-label-v5
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/fine_label_v5_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/fine_label_v5_%j.err

# =============================================================================
# fine-label-v5 — Etiqueta fina del corpus v5 usando MeSH (taxonomia v2:
# ~40 dominios eco+biomed+generales con regla de senal relativa).
# JOIN via duckdb/Parquet (30.8M filas, rapido) -> evita el indice SQLite lento.
# Reescribe train_corpus_v5.jsonl anadiendo mesh_terms + domain_fine
# SIN tocar text/orden => train_ids_v5.npy sigue siendo valido.
# CPU-only, Slurm (el script aborta en login node).
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"

python3 scripts/fine_label_v5.py \
    --input  "$BASE/data/train_corpus_v5.jsonl" \
    --db     /home/a474r867/work/pubmed/index/pubmed_fts.db \
    --parquet "$BASE/../litdump/pubmed/parsed/parquet/year=*/*.parquet" \
    --out    "$BASE/data/train_corpus_v5.jsonl" \
    --report "$BASE/data/v5_fine_report.json"

echo "fine_label_v5 done rc=$?"