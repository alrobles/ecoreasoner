#!/usr/bin/env bash
#SBATCH --job-name=unam-qa-fin
#SBATCH --partition=sixhour
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/unam_qa_fin_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/unam_qa_fin_%j.err
# Finaliza el pipeline QA->SFT de UNAM-EN (CPU only):
#   qa_filter --multiturn -> concat con qa_sft_v1 -> sft_pretok (npz)
# Env: IN_QA (dir qa_mt_shard*.jsonl), OUT_UNAM, OUT_ALL, OUT_NPZ, TOK,
#      V1JSONL (corpus a mezclar), MINREC, WORKERS
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
IN_QA=${IN_QA:-$BASE/data/qa_mt_unam}
OUT_UNAM=${OUT_UNAM:-$BASE/data/qa_sft_unam_mt.jsonl}
OUT_ALL=${OUT_ALL:-$BASE/data/qa_sft_v3_unam.jsonl}
OUT_NPZ=${OUT_NPZ:-$BASE/data/sft_ids_v3_unam.npz}
V1JSONL=${V1JSONL:-$BASE/data/qa_sft_v1.jsonl}
MINREC=${MINREC:-0.5}
WORKERS=${WORKERS:-8}
TOK=${TOK:-/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07}

export PYTHONPATH=/beegfs/a474r867/pydeps_tf451${PYTHONPATH:+:$PYTHONPATH}
export HF_HOME=/beegfs/a474r867/hf_cache
export HF_HUB_OFFLINE=1

echo "[fin] filter multiturn: $IN_QA -> $OUT_UNAM"
apptainer exec "$SIF" python3 "$BASE/scripts/qa_filter.py" \
  --multiturn --indir "$IN_QA" --out "$OUT_UNAM" --min-recall "$MINREC"

echo "[fin] concat unam + v1 -> $OUT_ALL"
apptainer exec "$SIF" python3 - "$OUT_UNAM" "$V1JSONL" "$OUT_ALL" <<'PY'
import sys
unam, v1, out = sys.argv[1:4]
n = m = 0
with open(out, "w", encoding="utf-8") as fo:
    for src in (unam, v1):
        try:
            for line in open(src, encoding="utf-8"):
                if line.strip():
                    fo.write(line); n += (src == unam); m += 1
        except FileNotFoundError:
            print(f"[fin] WARN {src} no existe — se omite")
print(f"[fin] unam={n} total={m} -> {out}")
PY

echo "[fin] pretok -> $OUT_NPZ"
apptainer exec "$SIF" python3 "$BASE/scripts/sft_pretok.py" \
  --input "$OUT_ALL" --tokenizer "$TOK" --out "$OUT_NPZ" \
  --workers "$WORKERS" --seq_len 768

echo "[fin] DONE"
ls -la "$OUT_NPZ" "$OUT_NPZ.meta.json"
