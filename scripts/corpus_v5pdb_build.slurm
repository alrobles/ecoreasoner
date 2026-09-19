#!/usr/bin/env bash
#SBATCH --job-name=cv5pdb
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/cv5pdb_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/cv5pdb_%j.err
# Corpus v5 para el 1B: papers_db (EN, min 200 chars) + mix esqueletos
# v4hneg -> corpus_v5_pdb.jsonl -> pretok --split-long -> train_ids_v5pdb.npz
# --split-long preserva fulltexts en ventanas 768 (no trunca tokens).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07
DB=$BASE/data/papersdb/papers_db.jsonl
SKEL=$BASE/data/corpus_v4hneg.jsonl
OUT=$BASE/data/corpus_v5_pdb.jsonl
NPZ=$BASE/data/train_ids_v5pdb.npz

echo "[$(hostname)] corpus_v5_pdb build $(date -u)"

# 1) filtrar papers_db EN + mezclar esqueletos
python3 $BASE/scripts/build_papersdb_corpus.py \
  --db "$DB" --out "$OUT" --skel "$SKEL" --min-chars 200

# 2) pretokenize con ventanas (preserva tokens de fulltexts)
apptainer exec "$SIF" python3 $BASE/scripts/pre_tokenize_v2.py \
  --input "$OUT" --tokenizer "$TOK" \
  --out "$NPZ" --workers 24 --seq_len 768 --split-long --read-block 150000
echo "=== V5PDB BUILD DONE $(date -u) ==="
