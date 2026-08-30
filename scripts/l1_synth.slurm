#!/usr/bin/env bash
#SBATCH --job-name=l1-synth
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/l1_synth_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/l1_synth_%j.err

# Sintetiza el dataset L1 (tool-use) desde las trazas B1.
# CPU-only. Escribe data/l1/train_corpus_l1.jsonl
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
python3 scripts/build_l1_synth.py \
  --input \
    "$BASE/data/distill_data.jsonl" \
    "$BASE/data/distill_v4_round2.jsonl" \
  --glob "$BASE/data/sci_v2_b1*.jsonl" \
  --out "$BASE/data/l1/train_corpus_l1.jsonl"
echo "l1-synth done rc=$?"