#!/bin/bash
#SBATCH --job-name=mdlm-moe-poc
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:mi210:1
#SBATCH --mem=64G
#SBATCH --time=05:50:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/mdlm_moe_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/mdlm_moe_%j.err

# =============================================================================
# dLLM-MoE PoC — masked-diffusion MoE, ola encadenada (patrón q35_wave).
# Uso: sbatch --export=WAVE_ID=poc0d5,TARGET_STEPS=N mdlm_moe_wave.slurm
# SIGUSR1 300s antes de walltime -> trainer guarda checkpoint + exit 42 ->
# este script relanza la siguiente ola (resume desde checkpoint-gN).
# =============================================================================
set -euo pipefail

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/qwen35-rocm-v2.sif
SCRIPTS=$BASE/scripts
DATA_DIR=$BASE/data
DATA="$DATA_DIR/sci_v2_b1.jsonl $DATA_DIR/sci_v2_existing.jsonl $DATA_DIR/sci_v1.jsonl"

WAVE_ID="${WAVE_ID:-poc0d5}"
OUT="${WAVE_OUT:-$BASE/outputs/$WAVE_ID}"
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
TARGET_STEPS="${TARGET_STEPS:-2000}"

mkdir -p "$OUT" "$BASE/logs"

echo "[$(date)] dLLM-MoE PoC $WAVE_ID empezando (TARGET_STEPS=$TARGET_STEPS)"

FINALIZED=0
finalize() {
    [ "$FINALIZED" = "1" ] && return
    FINALIZED=1
    if [ -f "$FLAG" ]; then echo "complete"; cat "$FLAG"; exit 0; fi
    if [ "$AUTO_RESUBMIT" = "1" ]; then
        echo "relanzando ola $WAVE_ID..."
        sbatch --export="AUTO_RESUBMIT=1,WAVE_ID=$WAVE_ID,WAVE_OUT=$OUT,TARGET_STEPS=$TARGET_STEPS" \
            "$BASE/scripts/mdlm_moe_wave.slurm" 2>/dev/null
    fi
}

handle_signal() {
    echo "=== SIGUSR1 $(date -u) — guardando ola ==="
    if [ -n "${TRAIN_PID:-}" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -USR1 "$TRAIN_PID" 2>/dev/null || true
        set +e; wait "$TRAIN_PID"; tx=$?; set -e
        echo "trainer exit $tx"
    fi
    finalize
}
trap handle_signal USR1
trap 'echo SIGTERM; exit 1' TERM

# ---- Config 0.5B MoE (hidden 768, 12 capas, 8 expertos top-1, ff_mult 4) ----
# total ~0.53B, active ~134M (25%). seq 768, mask_p 0.15
apptainer exec --nv \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$SIF" python3 "$SCRIPTS/train_mdlm_moe.py" \
        --data $DATA \
        --output "$OUT" \
        --hidden 768 --layers 12 --heads 12 \
        --n_experts 8 --expert_k 1 --ff_mult 4 \
        --seq_len 768 --batch_size 2 --grad_accum 2 \
        --lr 2e-4 --warmup 200 --mask_p 0.15 \
        --max_steps "$TARGET_STEPS" &
TRAIN_PID=$!
set +e; wait "$TRAIN_PID"; TRAIN_X=$?; set -e
echo "=== train exit $TRAIN_X ==="
case "$TRAIN_X" in
    0) echo "COMPLETE" > "$FLAG"; exit 0;;
    42) finalize;;
    *) if [ "${SIGNAL_RECEIVED:-0}" = "1" ]; then finalize; else exit "$TRAIN_X"; fi;;
esac
