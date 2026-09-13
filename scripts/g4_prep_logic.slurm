#!/bin/bash
#SBATCH --job-name=g4-prep-logic
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/g4_prep_logic_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/g4_prep_logic_%j.err
# G4 prep: corpus sintetico de consistencia fina (30K, 100% validos) ->
# train_ids_b4aug2logic.npz = skeleton_v2 + aug_b2_v2 + synth_logic.
set -u
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "FATAL: no es un job (guard login)"; exit 1; }

echo "[$(date)] 1/2 generar + pretokenizar synth_logic 30K"
apptainer exec "$SIF" python3 "$BASE/scripts/build_logic_synth.py" \
    --n 30000 --seed 7 \
    --out "$BASE/data/skeleton/train_corpus_synth_logic.jsonl"
apptainer exec "$SIF" python3 "$BASE/scripts/pre_tokenize_v2.py" \
    --input "$BASE/data/skeleton/train_corpus_synth_logic.jsonl" \
    --tokenizer "$TOK" --out "$BASE/data/train_ids_synth_logic.npz" \
    --workers 16 --seq_len 768

echo "[$(date)] 2/2 merge skeleton_v2 + augv2 + logic"
apptainer exec "$SIF" python3 - <<'EOF'
import json, time
import numpy as np
BASE = "/beegfs/a474r867/ecoreasoner/data"
parts = [np.load(f"{BASE}/train_ids_skeleton_v2.npz"),
         np.load(f"{BASE}/train_ids_augv2.npz"),
         np.load(f"{BASE}/train_ids_synth_logic.npz")]
m = int(parts[0]["vocab_size"]) if "vocab_size" in parts[0] else 126080
eos = int(parts[0]["eos_id"]) if "eos_id" in parts[0] else 0
ids = np.concatenate([p["ids"] for p in parts]).astype(np.int32)
lens = np.concatenate([p["lengths"] for p in parts]).astype(np.int32)
mx = int(ids.max())
assert mx < m, f"OOB_PROBLEMA max={mx} vocab={m}"
out = f"{BASE}/train_ids_b4aug2logic.npz"
np.savez(out, ids=ids, lengths=lens, vocab_size=m, eos_id=eos)
meta = {"n_tokens": int(ids.size), "n_docs": int(lens.size),
        "components": ["skeleton_v2", "augv2(11964)", "synth_logic(30K)"],
        "vocab_size": m, "eos_id": eos, "max_id": mx,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
with open(out + ".meta.json", "w") as f:
    json.dump(meta, f)
print(f"MERGE OK: {int(lens.size)} docs, {ids.size/1e9:.3f}B tok, "
      f"max_id {mx} < {m} (OOB_OK) -> {out}")
EOF

echo "[$(date)] G4 PREP LOGIC DONE"
