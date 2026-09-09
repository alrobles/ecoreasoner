#!/usr/bin/env bash
#SBATCH --job-name=dllm-v2-pairs
#SBATCH --partition=sixhour
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/dllm_v2_pairs_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/dllm_v2_pairs_%j.err
# =============================================================================
# Re-genera la bateria L0-L3 hard-negative (no re-tokeniza train/val).
# Asume que existen data/skeleton/train_skeleton_val.jsonl y el tokenizer LLaDA.
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
export HF_HUB_OFFLINE=1

mkdir -p "$BASE/runs/pairs_hard_v2" "$BASE/logs"

apptainer exec --overlay "$OVL":ro "$SIF" "$VENV/bin/python" "$BASE/harness/build_pairs_hard.py" \
    --in "$BASE/data/skeleton/train_skeleton_val.jsonl" \
    --tokenizer "$TOK" \
    --n 512 --max-len 768 \
    --out "$BASE/runs/pairs_hard_v2" --seed 7331

echo "=== dllm-v2-pairs DONE $(date -u) ==="
