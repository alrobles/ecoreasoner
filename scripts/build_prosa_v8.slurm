#!/usr/bin/env bash
#SBATCH --job-name=build-prosa-v8
#SBATCH --partition=sixhour
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/build_prosa_v8_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/build_prosa_v8_%j.err
# =============================================================================
# build_prosa_v8.slurm — pipeline CPU para construir prosa v8.
# No requiere GPU. Memoria alta para dedup y packing.
# Uso:
#   sbatch --export="LIMIT=10000" scripts/build_prosa_v8.slurm
# =============================================================================
set -euo pipefail
BASE="${BASE:-/beegfs/a474r867/ecoreasoner}"
PY="/opt/conda/bin/python"
LIMIT="${LIMIT:-0}"

echo "=== build_prosa_v8 LIMIT=$LIMIT $(date -u) ==="
"$PY" "$BASE/scripts/build_prosa_v8.py" \
    --config "$BASE/scripts/build_prosa_v8.yaml" \
    --limit-docs "$LIMIT"
echo "=== build_prosa_v8 done $(date -u) ==="
