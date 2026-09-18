#!/bin/bash
#SBATCH --job-name=consol_papers
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/consol_%j.out
#
# Consolidacion de papers -> data/papersdb/
#   STAGE=collect   streaming dedupe+limpieza (~50GB fuentes)
#   STAGE=skeleton  extrae esqueletos faltantes (16 workers)
#   STAGE=stats     reporte
# Uso: sbatch --export=ALL,STAGE=collect consol_papers.slurm
set -e
BASE=/beegfs/a474r867/ecoreasoner
STAGE=${STAGE:-collect}
OUT=${OUT:-$BASE/data/papersdb}
WORKERS=${WORKERS:-14}
cd $BASE
python3 scripts/consolidate_papers.py "$STAGE" --out "$OUT" --workers "$WORKERS"
echo "=== consol $STAGE DONE ==="
