#!/usr/bin/env bash
#SBATCH --job-name=l1-pretok
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/l1_pretok_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/l1_pretok_%j.err

# Pre-tokeniza el dataset L1 de tool-use + ancla v6 (mixto, 90/10) -> train_ids_l1.npy
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
cd "$BASE"
# 1) construir corpus mixto: L1 completo + muestra de v6 (ancla de conocimiento)
python3 - <<'EOF'
import json, random, os
random.seed(42)
l1 = []
with open("data/l1/train_corpus_l1.jsonl", encoding="utf-8") as f:
    for ln in f:
        if ln.strip():
            l1.append(json.loads(ln))
# ancla: v6 en proporcion ~10x (sampleo 1/50 del v6 = ~40K docs)
anchor = []
v6_path = "data/train_corpus_v6.jsonl"
if os.path.exists(v6_path):
    with open(v6_path, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if not ln.strip(): continue
            if i % 50 == 0:
                try:
                    anchor.append(json.loads(ln))
                except Exception:
                    pass
print(f"mix: l1={len(l1)} anchor={len(anchor)}")
out = l1 + anchor
random.shuffle(out)
with open("data/l1/train_corpus_l1mix.jsonl", "w", encoding="utf-8") as f:
    for d in out:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")
print("escrito", len(out), "docs")
EOF
# 2) pre-tokenizar el mix
apptainer exec "$SIF" python3 "$BASE/scripts/pre_tokenize.py" \
    --input "$BASE/data/l1/train_corpus_l1mix.jsonl" \
    --tokenizer "$TOK" \
    --out "$BASE/data/l1/train_ids_l1mix.npy" \
    --workers 16 --seq_len 768
# 3) verificar max < vocab (guard OOB)
apptainer exec "$SIF" python3 - <<'EOF'
import numpy as np
a = np.load("/beegfs/a474r867/ecoreasoner/data/l1/train_ids_l1mix.npy")
print("l1mix max:", int(a.max()), "min:", int(a.min()), "n:", a.size)
assert int(a.max()) < 126080, "OOB tokens!"
print("OOB_OK")
EOF
echo "l1-pretok done rc=$?"