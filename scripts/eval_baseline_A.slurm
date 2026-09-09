#!/bin/bash
#SBATCH --partition=sixhour
#SBATCH --time=00:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gres=gpu:mi210:2
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/eval_baseline_A-%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/eval_baseline_A-%j.err
#SBATCH --job-name=ecoreasoner-evalA
set -euo pipefail

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/qwen35-rocm-v2.sif
MODEL=$BASE/models/Qwen3.5-35B-A3B
SCRIPTS=$BASE/scripts
DATA=$BASE/data
OUT=$BASE/outputs

# Adaptadores de las 3 replicas
ADAPS="${OUT}/m1/final ${OUT}/w1/final ${OUT}/w2/final"
PROM=$DATA/prompts_toolcall_canonical.jsonl
MEM=$DATA/sci_v1_prompts.jsonl

mkdir -p $BASE/logs

# Copiar eval script si falta
if [ ! -f $SCRIPTS/eval_baseline_A.py ]; then
    echo "FATAL: $SCRIPTS/eval_baseline_A.py no existe"; exit 1
fi

echo "=== EVAL BASELINE A $(date -u) ===\n"
echo "=== lanzando eval dentro del SIF (3 replicas + base) ==="
apptainer exec --nv $SIF \
  python3 -u $SCRIPTS/eval_baseline_A.py \
  --base $MODEL \
  --adapters ${ADAPS} \
  --prompts $DATA/prompts_toolcall_canonical.jsonl \
  --mem-check $DATA/sci_v1_prompts.jsonl \
  --max-new-tokens 512 \
  --out $OUT/eval_baseline_A.json

echo "=== FIN $(date -u) ==="
ls -la $OUT/eval_baseline_A.json 2>/dev/null