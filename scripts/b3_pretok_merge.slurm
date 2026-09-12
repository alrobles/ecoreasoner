#!/bin/bash
#SBATCH --job-name=b3-pretok
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/b3_pretok_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/b3_pretok_%j.err
# B3: pre-tokeniza el corpus sintetico FLD (train_corpus_synth_b3.jsonl) y lo
# MEZCLA con el skeleton v2 (train_ids_skeleton_v2.npz, 299,534 docs, 123.2M tok)
# -> data/train_ids_b3.npz (~10% sintetico). Verificacion OOB obligatoria.
set -u
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
SYN=$BASE/data/skeleton/train_corpus_synth_b3.jsonl
OUT_SYN=$BASE/data/train_ids_synth_b3.npz
OUT_MIX=$BASE/data/train_ids_b3.npz
[[ -f "$SYN" ]] || { echo "FATAL: falta $SYN"; exit 1; }
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "FATAL: no es un job (guard login)"; exit 1; }

echo "[$(date)] pre-tokenizando sintetico -> $OUT_SYN"
apptainer exec "$SIF" python3 "$BASE/scripts/pre_tokenize_v2.py" \
    --input "$SYN" \
    --tokenizer "$TOK" \
    --out "$OUT_SYN" \
    --workers 16 --seq_len 768

echo "[$(date)] mezclando skeleton_v2 + synth -> $OUT_MIX"
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
        "vocab_size": m, "eos_id": eos, "max_id": mx,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
with open(mix_path + ".meta.json", "w") as f:
    json.dump(meta, f)
print(f"MERGE OK: {int(sk['lengths'].size)}+{int(sy['lengths'].size)} docs, "
      f"{ids.size/1e9:.3f}B tok, max_id {mx} < {m} (OOB_OK)")
EOF
echo "[$(date)] B3 PRETOK+MERGE DONE exit=$?"