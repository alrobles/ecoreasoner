#!/usr/bin/env bash
#SBATCH --job-name=cv4hneg
#SBATCH --partition=sixhour
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/cv4hneg_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/cv4hneg_%j.err
# G11 hneg-data: corpus_v4hneg = corpus_v4_en + hneg_pairs (pair-docs de
# knn_edges verificados). Luego pretokenize -> train_ids_v4hneg.npz.
# Cadena: build_hneg_pairs (CPU) -> merge -> pretok. Los brazos g11-hneg
# esperan el npz via soft-wait en g0_run.slurm (hasta 4h).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
V4EN=$BASE/data/corpus_v4_en.jsonl
HNEG=$BASE/data/hneg_pairs.jsonl
OUT=$BASE/data/corpus_v4hneg.jsonl

mkdir -p $BASE/logs
echo "[$(hostname)] corpus_v4hneg build $(date -u)"

# 1) minar pair-docs desde knn_edges
python3 $BASE/scripts/build_hneg_pairs.py --out "$HNEG"

# 2) merge: v4en + hneg
python3 - <<PY
n4 = nh = 0
with open("$OUT", "w") as fo:
    for line in open("$V4EN"):
        fo.write(line); n4 += 1
    for line in open("$HNEG"):
        fo.write(line); nh += 1
print(f"[v4hneg] v4en={n4} hneg={nh} total={n4+nh}", flush=True)
PY

# 3) pretokenize
apptainer exec "$SIF" python3 $BASE/scripts/pre_tokenize_v2.py \
  --input "$OUT" --tokenizer "$TOK" \
  --out $BASE/data/train_ids_v4hneg.npz --workers 16 --seq_len 768
echo "=== V4HNEG BUILD DONE $(date -u) ==="
