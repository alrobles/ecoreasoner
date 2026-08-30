#!/usr/bin/env bash
#SBATCH --job-name=arxiv-corpus
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/arxiv_corpus_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/arxiv_corpus_%j.err

# Une los shards full-text arXiv -> train_corpus_phys.jsonl (formato pipeline).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
python3 scripts/build_arxiv_corpus.py \
  --src "$BASE/data/arxiv/fulltext" \
  --out "$BASE/data/train_corpus_phys.jsonl" \
  --report "$BASE/data/arxiv_phys_report.json"
echo "arxiv-corpus done rc=$?"