#!/usr/bin/env bash
#SBATCH --job-name=arxiv-meta
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/arxiv_meta_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/arxiv_meta_%j.err

# Descarga el metadata OAI snapshot de arXiv (jackkuo HF, 4.58GB) y corre la
# seleccion balanceada de ~300K papers de fisica general.
# CPU-only. NO compite con bw1v5 (GPU).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
DIR="$BASE/data/arxiv"
mkdir -p "$DIR"
META="$DIR/arxiv-metadata-oai-snapshot.json"

if [ ! -f "$META" ]; then
  echo "[$(date)] descargando metadata arXiv (4.58GB)..."
  curl -L --fail --retry 3 -o "$META" \
    "https://huggingface.co/datasets/jackkuo/arXiv-metadata-oai-snapshot/resolve/main/arxiv-metadata-oai-snapshot.json"
  echo "[$(date)] metadata descargado: $(du -h "$META" | cut -f1)"
else
  echo "[$(date)] metadata ya existe: $(du -h "$META" | cut -f1)"
fi

echo "[$(date)] seleccion balanceada..."
python3 "$BASE/scripts/build_arxiv_select.py" \
  --meta "$META" \
  --out "$DIR/_arxiv_selected.jsonl" \
  --target 300000
echo "[$(date)] arxiv-meta DONE rc=$?"