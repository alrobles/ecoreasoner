#!/usr/bin/env bash
#SBATCH --job-name=f0-reeval
#SBATCH --partition=sixhour
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=60G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/f0_reeval_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/f0_reeval_%j.err
# =============================================================================
# F0 re-eval: corre suite_smoke + report sobre checkpoints YA entrenados de los
# micro-runs que completaron sin report.json (YAML roto / pairs ausente).
# NO re-entrena: solo carga el checkpoint y evalúa discriminación inferencial.
#
# TAGs: espacio-separados en $TAGS (default: los 3 prosa completados).
# Uso: sbatch --export="TAGS='f0-random-prosa f0-span-prosa f0-spanhi-prosa'" scripts/f0_reeval.slurm
# =============================================================================
set -uo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv/bin/python
RUNS=$BASE/runs
PAIRS=$RUNS/pairs.jsonl
TAGS="${TAGS:-f0-random-prosa f0-span-prosa f0-spanhi-prosa}"

for TAG in $TAGS; do
    OUT=$RUNS/$TAG
    CKPT_DIR=$(ls -d "$OUT"/checkpoint-g* 2>/dev/null | sort -t g -k2 -n | tail -1 || true)
    [ -z "$CKPT_DIR" ] && { echo "[$TAG] sin checkpoints"; continue; }
    CKPT="$CKPT_DIR/model.pt"
    CFG=$BASE/harness/configs/$TAG.yaml
    echo "=== [$TAG] eval $CKPT (pairs=$PAIRS) ==="
    apptainer exec --nv --overlay "$OVL":ro "$SIF" "$VENV/bin/python" \
        "$BASE/harness/suite_smoke.py" \
        --ckpt "$CKPT" --config "$CFG" --pairs "$PAIRS" \
        --out "$OUT/report.suite.json" --device cuda
    rc=$?
    echo "[$TAG] suite exit $rc"
    if [ $rc -eq 0 ]; then
        apptainer exec --nv --overlay "$OVL":ro "$SIF" "$VENV/bin/python" \
            "$BASE/harness/report.py" \
            --config "$CFG" --suite "$OUT/report.suite.json" \
            --out "$OUT/report.json" --job-id "$SLURM_JOB_ID" || echo "[$TAG] report FAILED"
    fi
    echo "=== [$TAG] -------"
done
echo "=== F0 REEVAL DONE $(date -u) ==="