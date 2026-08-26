#!/bin/bash
#SBATCH --job-name=pmc-fetch
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=05:50:00
#SBATCH --array=0-3
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/pmc_fetch_%A_%a.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/pmc_fetch_%A_%a.err

# =============================================================================
# PMC full-text masivo en paralelo (4 shards array). Descarga ~113K full-text
# nuevos desde pmc-oa-opendata S3 (24 workers c/u), write parcial por chunk.
# CPU-only: no compite con el retrain dLLM (GPU q6000).
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
CID=$SLURM_ARRAY_TASK_ID
echo "[$(date)] pmc-fetch shard $CID start"
python3 "$BASE/scripts/fetch_pmc_shard.py" \
  --pmids_file "$BASE/data/_new_pmids.txt" \
  --pmc_pkl "$BASE/data/_cand_pmc.pkl" \
  --chunk_id ${CID} --n_chunks 4 \
  --out_dir "$BASE/data" --workers 24
echo "[$(date)] pmc-fetch shard $CID DONE"