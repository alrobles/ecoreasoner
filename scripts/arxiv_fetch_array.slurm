#!/bin/bash
#SBATCH --job-name=arxiv-fetch
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=05:50:00
#SBATCH --array=0-7
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/arxiv_fetch_%A_%a.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/arxiv_fetch_%A_%a.err

# Descarga el source LaTeX de arXiv (e-print gzip) para los 300K papers
# seleccionados, 8 shards array, 24 workers c/u. CPU-only.
# Salidas: data/arxiv/fulltext/fulltext_c{id}.jsonl
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
CID=$SLURM_ARRAY_TASK_ID
NCH=8
echo "[$(date)] arxiv-fetch shard $CID start (of $NCH)"
python3 "$BASE/scripts/fetch_arxiv_text.py" \
  --sel "$BASE/data/arxiv/_arxiv_selected.jsonl" \
  --chunk_id ${CID} --n_chunks ${NCH} \
  --out_dir "$BASE/data/arxiv/fulltext" --workers 24 --min-text 1500
echo "[$(date)] arxiv-fetch shard $CID DONE"