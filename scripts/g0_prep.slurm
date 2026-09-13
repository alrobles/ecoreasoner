#!/bin/bash
#SBATCH --job-name=g0-prep
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/g0_prep_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/g0_prep_%j.err
# EvoG0 prep (CPU): (1) corpus synth marked-validity -> train_ids_b4m.npz,
# (2) dedup aug_b2 + merge -> train_ids_b4aug.npz, (3) holdout pairs n=2000.
set -u
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
SKEL=$BASE/data/train_ids_skeleton_v2.npz
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "FATAL: no es un job (guard login)"; exit 1; }

merge_npz() {  # $1=extra.npz $2=out.npz $3=meta_extra_json_string
apptainer exec "$SIF" python3 - "$SKEL" "$1" "$2" "$3" <<'EOF'
import json, sys, time
import numpy as np
sk_path, syn_path, mix_path, extra = sys.argv[1:5]
sk = np.load(sk_path); sy = np.load(syn_path)
m = sk["vocab_size"].item() if "vocab_size" in sk else 126080
eos = int(sk["eos_id"]) if "eos_id" in sk else 0
ids = np.concatenate([sk["ids"], sy["ids"]]).astype(np.int32)
lens = np.concatenate([sk["lengths"], sy["lengths"]]).astype(np.int32)
mx = int(ids.max())
assert mx < m, f"OOB_PROBLEMA max={mx} vocab={m}"
np.savez(mix_path, ids=ids, lengths=lens, vocab_size=m, eos_id=eos)
meta = {"n_tokens": int(ids.size), "n_docs": int(lens.size),
        "skeleton_docs": int(sk["lengths"].size), "extra_docs": int(sy["lengths"].size),
        "extra": extra, "vocab_size": m, "eos_id": eos, "max_id": mx,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
with open(mix_path + ".meta.json", "w") as f:
    json.dump(meta, f)
print(f"MERGE OK: {int(sk['lengths'].size)}+{int(sy['lengths'].size)} docs, "
      f"{ids.size/1e9:.3f}B tok, max_id {mx} < {m} (OOB_OK)")
EOF
}

echo "[$(date)] 1/3 corpus marked-validity"
apptainer exec "$SIF" python3 "$BASE/scripts/build_b3_synthetic.py" \
    --n 30000 --seed 42 --mark-validity \
    --out "$BASE/data/skeleton/train_corpus_synth_b4m.jsonl"
apptainer exec "$SIF" python3 "$BASE/scripts/pre_tokenize_v2.py" \
    --input "$BASE/data/skeleton/train_corpus_synth_b4m.jsonl" \
    --tokenizer "$TOK" --out "$BASE/data/train_ids_synth_b4m.npz" \
    --workers 16 --seq_len 768
merge_npz "$BASE/data/train_ids_synth_b4m.npz" "$BASE/data/train_ids_b4m.npz" \
    "synth_b4m 30K 70/30 marked"

echo "[$(date)] 2/3 dedup aug_b2 + merge"
apptainer exec "$SIF" python3 - <<'EOF'
import glob, json
seen, out = set(), open("/beegfs/a474r867/ecoreasoner/data/skeleton/train_aug_b2_merged.jsonl", "w")
n_in = n_out = 0
for fp in sorted(glob.glob("/beegfs/a474r867/ecoreasoner/data/skeleton/train_skeleton_aug_b2*.jsonl_s*")):
    for line in open(fp):
        n_in += 1
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("pid") in seen:
            continue
        seen.add(d["pid"]); n_out += 1
        out.write(json.dumps({"text": d["text"], "pid": d["pid"],
                              "domain": d.get("domain", "?")}) + "\n")
out.close()
print(f"AUG DEDUP: {n_in} lineas -> {n_out} docs unicos")
EOF
apptainer exec "$SIF" python3 "$BASE/scripts/pre_tokenize_v2.py" \
    --input "$BASE/data/skeleton/train_aug_b2_merged.jsonl" \
    --tokenizer "$TOK" --out "$BASE/data/train_ids_aug_b2.npz" \
    --workers 16 --seq_len 768
merge_npz "$BASE/data/train_ids_aug_b2.npz" "$BASE/data/train_ids_b4aug.npz" \
    "skeleton_v2 + aug_b2 dedup"

echo "[$(date)] 3/3 holdout pairs v4 (n=2000, seed 4242)"
apptainer exec "$SIF" python3 "$BASE/harness/build_pairs_hard_v3.py" \
    --in "$BASE/data/skeleton/train_skeleton_val.jsonl" \
    --species "$BASE/docs/results/species_eco_fine_tagged.json" \
    --tokenizer "$TOK" --n 2000 --seed 4242 \
    --out "$BASE/runs/pairs_hard_v4_holdout"

echo "[$(date)] G0 PREP DONE"
