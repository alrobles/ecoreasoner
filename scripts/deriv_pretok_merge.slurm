#!/bin/bash
#SBATCH --job-name=deriv-pretok
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/deriv_pretok_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/deriv_pretok_%j.err
# B2 (issue #9): pretok corpus_deriv_v1 + mezcla pareada.
#   train_ids_deriv.npz   : deriv pretokenizado (seq_len 768, LLaDA tok)
#   train_ids_b2base.npz  : subsample seedeado de v5pdb (~400M tok) — CONTROL
#   train_ids_b2deriv.npz : base + deriv upsampled a ~20% tokens — EXPERIMENTAL
set -u
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
DERIV=${DERIV_CORPUS:-$BASE/corpus/corpus_deriv_v1.jsonl}
OUT_DERIV=$BASE/data/train_ids_deriv.npz
BASE_TOK=${BASE_TOK:-400000000}
SHARE=${DERIV_SHARE:-0.20}
SEED=${MERGE_SEED:-7}
[[ -f "$DERIV" ]] || { echo "FATAL: falta $DERIV"; exit 1; }
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "FATAL: no es un job (guard login)"; exit 1; }

echo "[$(date)] pre-tokenizando deriv -> $OUT_DERIV"
apptainer exec "$SIF" python3 "$BASE/scripts/pre_tokenize_v2.py" \
    --input "$DERIV" \
    --tokenizer "$TOK" \
    --out "$OUT_DERIV" \
    --workers 16 --seq_len 768 || exit 1

echo "[$(date)] merge: subsample v5pdb ${BASE_TOK} tok + deriv share ${SHARE}"
apptainer exec "$SIF" python3 "$BASE/scripts/merge_deriv.py" \
    --v5pdb "$BASE/data/train_ids_v5pdb.npz" \
    --deriv "$OUT_DERIV" \
    --base-tokens "$BASE_TOK" --deriv-share "$SHARE" --seed "$SEED" \
    --base-out "$BASE/data/train_ids_b2base.npz" \
    --mix-out "$BASE/data/train_ids_b2deriv.npz" || exit 1

echo "[$(date)] DERIV PRETOK+MERGE DONE exit=$?"
