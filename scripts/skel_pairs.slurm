#!/usr/bin/env bash
#SBATCH --job-name=skel-pairs
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/skel_pairs_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/skel_pairs_%j.err

# F0: genera runs/pairs.jsonl (ctx/ok/bad en IDs) desde train_skeleton.jsonl
# para la eval de discriminación inferencial. Depende del FULL (afterok).
# CPU only.
set -uo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
mkdir -p "$BASE/runs"
echo "[$(date)] generando pares de discriminación..."
apptainer exec "$SIF" python3 "$BASE/harness/build_pairs.py" \
    --input "$BASE/data/skeleton/train_skeleton.jsonl" \
    --tokenizer "$TOK" \
    --out "$BASE/runs/pairs.jsonl" \
    --n-pairs 256 --seed 7331
echo "=== EXIT $? ==="