#!/usr/bin/env bash
#SBATCH --job-name=curate-val
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/curate_val_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/curate_val_%j.err

# Valida el pipeline de curacion por conectividad en muestra (n_per_domain=5000).
# CPU-only (embeddings BGE-small en CPU). Calibra umbrales para la curacion final.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
# token HF local no existe en HPC; esperamos HF_TOKEN en env (si no, modelos publicos)
export HF_HOME=/beegfs/a474r867/hf-cache
"$BASE/venv_curate/bin/python" "$BASE/scripts/curate_validate.py" \
  --v6 "$BASE/data/train_corpus_v6.jsonl" \
  --phys "$BASE/data/train_corpus_phys.jsonl" \
  --n-per-domain 5000 --k 20 \
  --out "$BASE/data/curate/validate_report.json"
echo "curate-val done rc=$?"