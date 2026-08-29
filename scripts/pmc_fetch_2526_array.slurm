#!/bin/bash
#SBATCH --job-name=pmc-fetch2526
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=05:50:00
#SBATCH --array=0-7
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/pmc_fetch2526_%A_%a.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/pmc_fetch2526_%A_%a.err

# =============================================================================
# PMC full-text SHARD 2025-2026 (los 2 anios mas recientes NO cubiertos por v5).
# 526,735 pmids nuevos (dedup vs pmc_corpus_v4). 8 shards array, 16 workers c/u.
# 16 cpus/shard (128 total) para programar en el pool CPU ocupado del cluster.
# CPU-only: no compite con bw1v5 (GPU Blackwell).
# Salidas parciales: data/fulltext_2526/fulltext_corpus_c{id}.jsonl
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
CID=$SLURM_ARRAY_TASK_ID
NCH=8
OUTDIR="$BASE/data/fulltext_2526"
mkdir -p "$OUTDIR"
echo "[$(date)] pmc-fetch2526 shard $CID start (of $NCH)"
python3 "$BASE/scripts/fetch_pmc_shard.py" \
  --pmids_file "$BASE/data/_new_pmids_2024_2025.txt" \
  --pmc_pkl "$BASE/data/_cand_pmc_2024_2025.pkl" \
  --chunk_id ${CID} --n_chunks ${NCH} \
  --out_dir "$OUTDIR" --workers 16
echo "[$(date)] pmc-fetch2526 shard $CID DONE"
