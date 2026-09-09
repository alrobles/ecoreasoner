#!/usr/bin/env bash
#SBATCH --job-name=ceiling-llada8b
#SBATCH --partition=sixhour
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:pro6000:1
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/ceiling_llada8b_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/ceiling_llada8b_%j.err
# =============================================================================
# ceiling_llada8b.slurm — techo LLaDA-8B-Base en suite_smoke.
# Requiere pro6000 (96GB) por el tamano del modelo en bf16.
# Uso:
#   sbatch --export="PAIRS=/beegfs/.../runs/pairs_hard/pairs_L0.jsonl" \
#       scripts/ceiling_llada8b.slurm
# =============================================================================
set -euo pipefail
BASE="${BASE:-/beegfs/a474r867/ecoreasoner}"
SIF="${SIF:-$BASE/pytorch-cuda.sif}"
OVL="${OVL:-$BASE/blackwell-torch.overlay}"
PY="/bb/bwvenv/bin/python"
PAIRS="${PAIRS:-$BASE/runs/pairs_hard/pairs_L0.jsonl}"
PAIRS_TOKENIZER="${PAIRS_TOKENIZER:-$BASE/tokenizer/LLaDA}"
OUT="${OUT:-$BASE/runs/ceiling_llada8b.json}"
MC_NUM="${MC_NUM:-32}"

echo "=== ceiling_llada8b PAIRS=$PAIRS TOKENIZER=$PAIRS_TOKENIZER $(date -u) ==="
apptainer exec --nv --overlay "$OVL":ro "$SIF" "$PY" \
    "$BASE/scripts/ceiling_llada8b.py" \
    --pairs "$PAIRS" --pairs-tokenizer "$PAIRS_TOKENIZER" \
    --out "$OUT" --device cuda --mc-num "$MC_NUM"
echo "=== ceiling_llada8b done $(date -u) ==="
