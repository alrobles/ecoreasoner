#!/usr/bin/env bash
#SBATCH --job-name=cv4en
#SBATCH --partition=sixhour
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/cv4en_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/cv4en_%j.err
# corpus_v4_en = corpus_v3_en (c1e) + UNAM-EN jsonl. Luego pretokenize ->
# train_ids_v4en.npz. Corre cuando la traducción termina (bandera).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
UNAM_MD=${UNAM_MD:-/beegfs/a474r867/unam_md_en}
V3EN=$BASE/data/corpus_v3_en.jsonl
OUT=$BASE/data/corpus_v4_en.jsonl
UNAM_JL=$BASE/data/unam_en.jsonl

mkdir -p $BASE/logs
echo "[$(hostname)] corpus_v4_en build"

python3 $BASE/scripts/unam_en_to_jsonl.py --indir "$UNAM_MD" --out "$UNAM_JL"

python3 - <<PY
import json, random
n3 = nu = 0
with open("$OUT", "w") as fo:
    for line in open("$V3EN"):
        fo.write(line); n3 += 1
    for line in open("$UNAM_JL"):
        fo.write(line); nu += 1
print(f"[v4en] v3en={n3} unam={nu} total={n3+nu}", flush=True)
PY

apptainer exec "$SIF" python3 $BASE/scripts/pre_tokenize_v2.py \
  --input "$OUT" --tokenizer "$TOK" \
  --out $BASE/data/train_ids_v4en.npz --workers 16 --seq_len 768
echo "=== V4EN BUILD DONE ==="
