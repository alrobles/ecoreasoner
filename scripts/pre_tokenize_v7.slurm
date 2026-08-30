#!/usr/bin/env bash
#SBATCH --job-name=v7-pretok
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/v7_pretok_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/v7_pretok_%j.err

# Pre-tokenizacion del corpus CURADO v7 (train_corpus_v7_curated.jsonl,
# ~1.1M docs full-text, ~3.9B tok) -> train_ids_v7.npy.
# CPU-only (igual que v5/v6/phys). Verifica OOB al final (regla obligatoria).
set -u
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
echo "[$(date)] pre-tokenizando train_corpus_v7_curated -> train_ids_v7.npy"
apptainer exec "$SIF" python3 "$BASE/scripts/pre_tokenize.py" \
    --input "$BASE/data/curate/train_corpus_v7_curated.jsonl" \
    --tokenizer "$TOK" \
    --out "$BASE/data/train_ids_v7.npy" \
    --workers 16 --seq_len 768
echo "=== verificar OOB (max < vocab) ==="
apptainer exec "$SIF" python3 - <<'EOF'
import numpy as np
a = np.load("/beegfs/a474r867/ecoreasoner/data/train_ids_v7.npy")
print("v7 max:", int(a.max()), "min:", int(a.min()), "n:", a.size)
print("OOB_OK" if int(a.max()) < 126080 else "OOB_PROBLEMA (clamp necesario)")
EOF
echo "[$(date)] PRE-TOKENIZE v7 DONE exit=$?"