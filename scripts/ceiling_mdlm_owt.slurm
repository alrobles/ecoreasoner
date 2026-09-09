#!/usr/bin/env bash
#SBATCH --job-name=ceiling-mdlm-owt
#SBATCH --partition=sixhour
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:pro6000:1
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/ceiling_mdlm_owt_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/ceiling_mdlm_owt_%j.err
# =============================================================================
# ceiling_mdlm_owt.slurm — techo MDLM-OWT (~130M) en suite_smoke.
# Uso:
#   sbatch --export="PAIRS=/beegfs/.../runs/pairs_hard/pairs_L0.jsonl" \
#       scripts/ceiling_mdlm_owt.slurm
# =============================================================================
set -euo pipefail
BASE="${BASE:-/beegfs/a474r867/ecoreasoner}"
SIF="${SIF:-$BASE/pytorch-cuda.sif}"
OVL="${OVL:-$BASE/blackwell-torch.overlay}"
PY="/bb/bwvenv/bin/python"
PAIRS="${PAIRS:-$BASE/runs/pairs_hard/pairs_L0.jsonl}"
PAIRS_TOKENIZER="${PAIRS_TOKENIZER:-$BASE/tokenizer/LLaDA}"
OUT="${OUT:-$BASE/runs/ceiling_mdlm_owt.json}"

echo "=== ceiling_mdlm_owt PAIRS=$PAIRS TOKENIZER=$PAIRS_TOKENIZER $(date -u) ==="
apptainer exec --nv --overlay "$OVL":ro "$SIF" "$PY" \
    "$BASE/scripts/ceiling_mdlm_owt.py" \
    --pairs "$PAIRS" --pairs-tokenizer "$PAIRS_TOKENIZER" \
    --out "$OUT" --device cuda
echo "=== ceiling_mdlm_owt done $(date -u) ==="
