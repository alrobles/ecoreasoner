#!/bin/bash
#SBATCH --job-name=pmc-fulltext
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/pmc_fulltext_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/pmc_fulltext_%j.err

# =============================================================================
# PMC full-text a escala: descripta los ~15K docs con PMC del eco_corpus
# via pmc-oa-opendata.s3.amazonaws.com (24 workers). Sin FTP/captcha.
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
echo "[$(date)] PMC fulltext a escala (24 workers)"
python3 "$BASE/scripts/map_pmc_fulltext.py" \
    --pmids_corpus "$BASE/data/eco_corpus.jsonl" \
    --ids "/beegfs/a474r867/litdump/pubmed/PMC-ids.csv.gz" \
    --out "$BASE/data/fulltext_corpus.jsonl" \
    --limit 20000 \
    --max_workers 24
echo "[$(date)] DONE"