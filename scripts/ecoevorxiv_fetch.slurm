#!/bin/bash
#SBATCH --job-name=ecoevorxiv-fetch
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=05:50:00
#SBATCH --array=0-3
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/ecoevorxiv_fetch_%A_%a.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/ecoevorxiv_fetch_%A_%a.err

# =============================================================================
# Descarga full-text de EcoEvoRxiv (PDF -> pdfplumber) en paralelo (4 shards).
# CPU-only: no compite con el retrain dLLM (GPU q6000).
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
CID=$SLURM_ARRAY_TASK_ID
echo "[$(date)] ecoevorxiv-fetch shard $CID start"
python3 "$BASE/scripts/fetch_ecoevorxiv.py" \
  --meta "$BASE/data/ecoevorxiv_meta.jsonl" \
  --chunk_id ${CID} --n_chunks 4 \
  --out_dir "$BASE/data" --tmp "$BASE/tmp_eev" --workers 8
echo "[$(date)] ecoevorxiv-fetch shard $CID DONE"