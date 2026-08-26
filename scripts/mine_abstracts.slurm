#!/bin/bash
#SBATCH --job-name=mine-abstracts
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/mine_abstracts_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/mine_abstracts_%j.err

# =============================================================================
# Re-mina abstracts PubMed (DuckDB sobre Parquet) a gran escala (~315K eco
# + dominios phylo/genom/bioc) para ampliar la cobertura del corpus base.
# CPU-only multi-dominio, estratificado por década.
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
echo "[$(date)] mine abstracts a escala"
python3 "$BASE/scripts/mine_pubmed_duckdb.py" \
  --out "$BASE/data/eco_corpus_v2.jsonl" \
  --max_per_domain 250000 \
  --min_year 1990 --min_abstract_len 200 \
  --domains eco,phylo,genom,bioc
echo "[$(date)] DONE"