#!/usr/bin/env bash
#SBATCH --job-name=skel-pretok
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=05:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/skel_pretok_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/skel_pretok_%j.err

# Pre-tokenizacion del corpus de ESQUELETOS (train_skeleton.jsonl,
# ~297K docs con >=3 etapas, salida del F0b full) -> train_ids_skeleton.npy.
# CPU-only (patron pre_tokenize_v7). Verifica OOB al final (regla obligatoria:
# max < vocab, vocab real = 126080; si OOB -> clamp necesario antes de entrenar).
set -u
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
echo "[$(date)] pre-tokenizando train_skeleton.jsonl -> train_ids_skeleton.npy"
apptainer exec "$SIF" python3 "$BASE/scripts/pre_tokenize.py" \
    --input "$BASE/data/skeleton/train_skeleton.jsonl" \
    --tokenizer "$TOK" \
    --out "$BASE/data/train_ids_skeleton.npy" \
    --workers 16 --seq_len 768
echo "=== verificar OOB (max < vocab) ==="
apptainer exec "$SIF" python3 - <<'EOF'
import numpy as np
a = np.load("/beegfs/a474r867/ecoreasoner/data/train_ids_skeleton.npy")
print("skeleton max:", int(a.max()), "min:", int(a.min()), "n:", a.size)
print("OOB_OK" if int(a.max()) < 126080 else "OOB_PROBLEMA (clamp necesario)")
EOF
echo "[$(date)] PRE-TOKENIZE skeleton DONE exit=$?"