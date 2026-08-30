#!/usr/bin/env bash
#SBATCH --job-name=curate-final
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=96G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/curate_final_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/curate_final_%j.err

# Cura el corpus COMPLETO (v6 + phys, ~2.2M docs) por CONECTIVIDAD CONCEPTUAL.
# Embeddings BGE-small en GPU (1x pro6000) -> kNN -> dedup sem -> top-frac% por dominio.
# GPU: 2.2M docs ~1-2h. Escribe data/curate/train_corpus_v7_curated.jsonl
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
export HF_HOME=/beegfs/a474r867/hf-cache
export CUDA_VISIBLE_DEVICES=0
"$BASE/venv_curate/bin/python" "$BASE/scripts/curate_final.py" \
  --v6 "$BASE/data/train_corpus_v6.jsonl" \
  --phys "$BASE/data/train_corpus_phys.jsonl" \
  --out "$BASE/data/curate/train_corpus_v7_curated.jsonl" \
  --report "$BASE/data/curate/curate_final_report.json" \
  --vecs "$BASE/data/curate/vectors_full.npy" \
  --k 20 --dup-cos 0.95 --frac 0.5 --batch 1024
echo "curate-final done rc=$?"