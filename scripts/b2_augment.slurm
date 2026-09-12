#!/bin/bash
#SBATCH --job-name=b2-augment
#SBATCH --partition=sixhour
#SBATCH --time=05:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/b2_augment_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/b2_augment_%j.err
# B2: genera HIPOTESIS+PREDICCION con DeepSeek-V4-Flash (ollama-v4serve).
# Resume via .done; auto-resubmit hasta completar N docs.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
IN=${IN:-$BASE/data/skeleton/train_skeleton_train.jsonl}
OUT=${OUT:-$BASE/data/skeleton/train_skeleton_aug_b2.jsonl}
N=${N:-40000}
MODEL=${MODEL:-deepseek-v4-flash}
SHARD=${SHARD:-0}
NSHARDS=${NSHARDS:-1}
OUTFILE="${OUT}_s${SHARD}"

echo "=== b2-augment $(date -u) in=$IN out=$OUTFILE n=$N shard=$SHARD/$NSHARDS ==="

# esperar a que haya un v4serve RUNNING (max 20 min)
for i in $(seq 1 40); do
    if squeue -u "$USER" -n ollama-v4serve -t R -h | grep -q .; then
        echo "v4serve activo"; break
    fi
    echo "esperando v4serve... ($i)"; sleep 30
done

python3 -u "$BASE/scripts/augment_inferential_stage.py" \
    --input "$IN" --out "$OUTFILE" --n "$N" --model "$MODEL" \
    --shard "$SHARD" --nshards "$NSHARDS" || true

# resubmit solo si murió sin llegar a COMPLETE/EXHAUSTED (p.ej. límite de 6h)
if ! tail -3 /beegfs/a474r867/ecoreasoner/logs/b2_augment_${SLURM_JOB_ID}.out 2>/dev/null \
        | grep -qE "COMPLETE|EXHAUSTED"; then
    echo "murio sin terminar -> resubmit"
    sbatch --export=ALL,IN,OUT,N,MODEL,SHARD,NSHARDS "$0"
else
    echo "=== b2-augment fin $(date -u) ==="
fi
