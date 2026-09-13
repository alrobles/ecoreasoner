#!/bin/bash
#SBATCH --job-name=b4-pretok
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/b4_pretok_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/b4_pretok_%j.err
# B4 (ABLACION): igual que B3 pero corpus sintetico 100% VALIDO (sin near-miss).
# Test: si number/negation se reparan, los near-miss como positivos de denoising
# eran el supresor. Pretok train_corpus_synth_b4.jsonl + merge con skeleton_v2.
set -u
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
SYN=$BASE/data/skeleton/train_corpus_synth_b4.jsonl
OUT_SYN=$BASE/data/train_ids_synth_b4.npz
OUT_MIX=$BASE/data/train_ids_b4.npz
[[ -f "$SYN" ]] || { echo "FATAL: falta $SYN"; exit 1; }
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "FATAL: no es un job (guard login)"; exit 1; }

echo "[$(date)] pre-tokenizando sintetico B4 -> $OUT_SYN"
apptainer exec "$SIF" python3 "$BASE/scripts/pre_tokenize_v2.py" \
    --input "$SYN" \
    --tokenizer "$TOK" \
    --out "$OUT_SYN" \
    --workers 16 --seq_len 768

echo "[$(date)] mezclando skeleton_v2 + synth_b4 -> $OUT_MIX"
apptainer exec "$SIF" python3 - "$OUT_SYN" "$OUT_MIX" <<'EOF'
import json, sys, time
import numpy as np
syn_path, mix_path = sys.argv[1], sys.argv[2]
sk = np.load("/beegfs/a474r867/ecoreasoner/data/train_ids_skeleton_v2.npz")
sy = np.load(syn_path)
m = sk["vocab_size"].item() if "vocab_size" in sk else 126080
eos = int(sk["eos_id"]) if "eos_id" in sk else 0
ids = np.concatenate([sk["ids"], sy["ids"]]).astype(np.int32)
lens = np.concatenate([sk["lengths"], sy["lengths"]]).astype(np.int32)
mx = int(ids.max())
assert mx < m, f"OOB_PROBLEMA max={mx} vocab={m}"
np.savez(mix_path, ids=ids, lengths=lens, vocab_size=m, eos_id=eos)
meta = {"n_tokens": int(ids.size), "n_docs": int(lens.size),
        "skeleton_docs": int(sk["lengths"].size), "synth_docs": int(sy["lengths"].size),
        "synth_valid_ratio": 1.0,
        "vocab_size": m, "eos_id": eos, "max_id": mx,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
with open(mix_path + ".meta.json", "w") as f:
    json.dump(meta, f)
print(f"MERGE OK: {int(sk['lengths'].size)}+{int(sy['lengths'].size)} docs, "
      f"{ids.size/1e9:.3f}B tok, max_id {mx} < {m} (OOB_OK)")
EOF
echo "[$(date)] B4 PRETOK+MERGE DONE exit=$?"
