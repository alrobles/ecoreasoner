#!/usr/bin/env bash
#SBATCH --job-name=pmc-ingest
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-19
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/pmc_ingest_%A_%a.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/pmc_ingest_%A_%a.err

# Slurm array: ingesta PMC OA 2024-2026 (shards). Cada tarea procesa su slice.
# Requiere: pmc_shards_<N>/shard_<i>.txt (lista de PMCIDs por array task), generado por gen_shards.py

set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SHARDS=$BASE/data/pmc_shards_20
OUT=$BASE/data/pmc_parquet
mkdir -p "$OUT"

LIST="$SHARDS/shard_${SLURM_ARRAY_TASK_ID}.txt"
[ -f "$LIST" ] || { echo "no shard list $LIST"; exit 1; }

echo "[$(date)] shard $SLURM_ARRAY_TASK_ID: $(wc -l < "$LIST") PMCIDs"
cd /home/a474r867
python3 pmc_worker.py \
  --list-file "$LIST" \
  --anio-min 2024 \
  --out "$OUT/shard_${SLURM_ARRAY_TASK_ID}.parquet"
echo "[$(date)] shard $SLURM_ARRAY_TASK_ID done"