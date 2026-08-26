#!/bin/bash
#SBATCH --job-name=mdlm-retrain
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:q6000:4
#SBATCH --mem=150G
#SBATCH --time=05:50:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/mdlm_retrain_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/mdlm_retrain_%j.err

# =============================================================================
# dLLM-MoE RETRAIN — masked-diffusion MoE sobre el CORPUS GRANDE
#   (train_corpus.jsonl, 44,045 docs / ~165M tokens)
# 4x Q6000 DDP (CUDA/pytorch-cuda.sif), top-1 MoE 8 exp, 676M/279M activos.
# Uso: sbatch --export=WAVE_ID=retrain1,TARGET_STEPS=N mdlm_retrain.slurm
# SIGUSR1 300s antes de walltime -> checkpoint + exit 42 -> relanza (ola).
# =============================================================================
set -euo pipefail

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
SCRIPTS=$BASE/scripts
DATA_DIR=$BASE/data
DATA="$DATA_DIR/train_corpus_v2.jsonl"

WAVE_ID="${WAVE_ID:-retrain1}"
OUT="${WAVE_OUT:-$BASE/outputs/$WAVE_ID}"
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
TARGET_STEPS="${TARGET_STEPS:-5000}"

mkdir -p "$OUT" "$BASE/logs"
echo "[$(date)] dLLM-MoE RETRAIN $WAVE_ID (4x Q6000 DDP, TARGET_STEPS=$TARGET_STEPS)"

FINALIZED=0
finalize() {
    [ "$FINALIZED" = "1" ] && return
    FINALIZED=1
    if [ -f "$FLAG" ]; then echo "complete"; cat "$FLAG"; exit 0; fi
    if [ "$AUTO_RESUBMIT" = "1" ]; then
        echo "relanzando ola $WAVE_ID..."
        sbatch --export="AUTO_RESUBMIT=1,WAVE_ID=$WAVE_ID,WAVE_OUT=$OUT,TARGET_STEPS=$TARGET_STEPS" \
            "$BASE/scripts/mdlm_retrain.slurm" 2>/dev/null
    fi
}
handle_signal() {
    echo "=== SIGUSR1 $(date -u) — guardando ola ==="
    if [ -n "${SRUN_PID:-}" ] && kill -0 "$SRUN_PID" 2>/dev/null; then
        kill -USR1 "$SRUN_PID" 2>/dev/null || true
        set +e; wait "$SRUN_PID"; tx=$?; set -e
    fi
    finalize
}
trap handle_signal USR1
trap 'echo SIGTERM; exit 1' TERM

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29513
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# 0.5B MoE, batch 2, grad_accum 2, mask_p 0.15 (igual que PoC, corpus grande)
srun -n 4 /usr/bin/apptainer exec --nv \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$SIF" python3 "$SCRIPTS/train_mdlm_moe.py" \
        --data $DATA --output "$OUT" \
        --hidden 768 --layers 12 --heads 12 \
        --n_experts 8 --expert_k 1 --ff_mult 4 \
        --seq_len 768 --batch_size 2 --grad_accum 2 \
        --lr 2e-4 --warmup 200 --mask_p 0.15 \
        --max_steps "$TARGET_STEPS" &
SRUN_PID=$!
set +e; wait "$SRUN_PID"; TRAIN_X=$?; set -e
echo "=== srun exit $TRAIN_X ==="
case "$TRAIN_X" in
    0) echo "COMPLETE" > "$FLAG"; exit 0;;
    42) finalize;;
    *) if [ "${SIGNAL_RECEIVED:-0}" = "1" ]; then finalize; else exit "$TRAIN_X"; fi;;
esac