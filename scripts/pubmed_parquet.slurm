#!/bin/bash
#SBATCH --job-name=pubmed-parquet
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/pubmed_parquet_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/pubmed_parquet_%j.err

# =============================================================================
# Port PubMed SQLite (30.8M rows) -> Parquet particionado por año (DuckDB)
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
echo "[$(date)] port pubmed -> parquet (DuckDB sqlite_scan)"
python3 "$BASE/scripts/port_pubmed_parquet.py"
echo "[$(date)] DONE port"