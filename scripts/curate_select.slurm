#!/usr/bin/env bash
#SBATCH --job-name=curate-sel
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/curate_sel_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/curate_sel_%j.err

# FASE 2: kNN + dedup semantico + seleccion top-frac por dominio (lee vectors)
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
"$BASE/venv_curate/bin/python" "$BASE/scripts/curate_final.py" \
  --v6 "$BASE/data/train_corpus_v6.jsonl" \
  --phys "$BASE/data/train_corpus_phys.jsonl" \
  --vecs "$BASE/data/curate/vectors_full.npy" \
  --out "$BASE/data/curate/train_corpus_v7_curated.jsonl" \
  --report "$BASE/data/curate/curate_final_report.json" \
  --k 20 --dup-cos 0.95 --frac 0.5 --from-vecs
echo "curate-sel done rc=$?"