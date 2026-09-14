#!/usr/bin/env bash
#SBATCH --job-name=c1ebuild
#SBATCH --partition=sixhour
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/c1ebuild_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/c1ebuild_%j.err
# c1e: corpus v3 SOLO-EN (sin chunks UNAM-es) — ablacion para aislar si la
# caida de g8-c1 (L3 0.568 vs ~0.63 backbone) viene de la dilucion ES (11.7%
# del corpus). Filtra corpus_v3.jsonl -> corpus_v3_en.jsonl + pretokenize.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
python3 - <<'PY'
import json
inp = "/beegfs/a474r867/ecoreasoner/data/corpus_v3.jsonl"
out = "/beegfs/a474r867/ecoreasoner/data/corpus_v3_en.jsonl"
kept = dropped = 0
with open(inp) as fi, open(out, "w") as fo:
    for line in fi:
        d = json.loads(line)
        if d.get("lang") == "es" or str(d.get("domain", "")).startswith("unam"):
            dropped += 1
            continue
        fo.write(line)
        kept += 1
print(f"[c1e] kept={kept} dropped_es={dropped}", flush=True)
PY
apptainer exec $BASE/pytorch-cuda.sif python3 $BASE/scripts/pre_tokenize_v2.py \
  --input $BASE/data/corpus_v3_en.jsonl --tokenizer $TOK \
  --out $BASE/data/train_ids_c1e.npz --workers 16 --seq_len 768
echo "=== C1E BUILD DONE ==="
