#!/usr/bin/env bash
#SBATCH --job-name=llada-moe-dl
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/llada_moe_dl_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/llada_moe_dl_%j.err

# Descarga de inclusionAI/LLaDA-MoE-7B-A1B-Instruct (Apache 2.0) a hf-cache
# + clone del repo oficial LLaDA (train.sh --finetune_from_hf) para el
# fine-tune de tool-use ecologico sobre un base YA entrenado (pivote 2026-08-30).
set -euo pipefail
export HF_HOME=/beegfs/a474r867/hf-cache
export HF_HUB_ENABLE_HF_TRANSFER=0
echo "[$(date)] descargando inclusionAI/LLaDA-MoE-7B-A1B-Instruct..."
python3 - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("inclusionAI/LLaDA-MoE-7B-A1B-Instruct")
print("OK:", p)
EOF
echo "[$(date)] clonando repo LLaDA (fine-tune)..."
[ -d /beegfs/a474r867/LLaDA ] || git clone --depth 1 https://github.com/ML-GSAI/LLaDA.git /beegfs/a474r867/LLaDA
echo "[$(date)] LLADA-MOE-DL DONE exit=$?"