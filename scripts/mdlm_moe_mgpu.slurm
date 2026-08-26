#!/bin/bash
#SBATCH --job-name=mdlm-moe-mgpu
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:q6000:4
#SBATCH --mem=120G
#SBATCH --time=05:50:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/mdlm_moe_mgpu_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/mdlm_moe_mgpu_%j.err

# =============================================================================
# dLLM-MoE PoC — multi-GPU masked-diffusion MoE con DDP (4x Q6000 en 1 nodo).
# Uso: sbatch --export=WAVE_ID=poc0d5,TARGET_STEPS=N mdlm_moe_mgpu.slurm
# DDP: srun lanza 4 ranks (uno por Q6000), gradientes promediados via nccl,
# ~4x throughput vs single-GPU. Cada rank entrena un slice distinto de batches.
# SIGUSR1 300s antes de walltime -> guarda checkpoint + exit 42 -> relanza.
# =============================================================================
set -euo pipefail

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
SCRIPTS=$BASE/scripts
DATA_DIR=$BASE/data
DATA="$DATA_DIR/sci_v2_b1.jsonl $DATA_DIR/sci_v2_existing.jsonl $DATA_DIR/sci_v1.jsonl"

WAVE_ID="${WAVE_ID:-poc0d5mg}"
OUT="${WAVE_OUT:-$BASE/outputs/$WAVE_ID}"
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
TARGET_STEPS="${TARGET_STEPS:-2000}"

mkdir -p "$OUT" "$BASE/logs"
echo "[$(date)] dLLM-MoE multi-GPU $WAVE_ID (4x Q6000 DDP, TARGET_STEPS=$TARGET_STEPS)"

FINALIZED=0
finalize() {
    [ "$FINALIZED" = "1" ] && return
    FINALIZED=1
    if [ -f "$FLAG" ]; then echo "complete"; cat "$FLAG"; exit 0; fi
    if [ "$AUTO_RESUBMIT" = "1" ]; then
        echo "relanzando ola $WAVE_ID..."
        sbatch --export="AUTO_RESUBMIT=1,WAVE_ID=$WAVE_ID,WAVE_OUT=$OUT,TARGET_STEPS=$TARGET_STEPS" \
            "$BASE/scripts/mdlm_moe_mgpu.slurm" 2>/dev/null
    fi
}
handle_signal() {
    echo "=== SIGUSR1 $(date -u) — guardando ola ==="
    # kill srun to trigger DDP checkpoint on each rank
    if [ -n "${SRUN_PID:-}" ] && kill -0 "$SRUN_PID" 2>/dev/null; then
        kill -USR1 "$SRUN_PID" 2>/dev/null || true
        set +e; wait "$SRUN_PID"; tx=$?; set -e
    fi
    finalize
}
trap handle_signal USR1
trap 'echo SIGTERM; exit 1' TERM

# ---- DDP rendezvous env (torch uses env://) ----
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29512
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Config 0.5B MoE, batch 2, grad_accum 2, mask_p 0.15
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