#!/bin/bash
#SBATCH --job-name=v5-1b-mi210
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:mi210:3
#SBATCH --constraint="mi210"
#SBATCH --mem=170G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/v5_1b_mi210_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/v5_1b_mi210_%j.err
# V5-1B sobre pool AMD MI210 (gfx90a, 64GB) — variante de v5_1b_ddp.slurm.
# Mismo trainer/args/ckpt; cambia el stack: venv_rocm (torch 2.5.1+rocm6.2,
# SIN apptainer) + torchrun intra-nodo + RCCL via SHM
# (NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 — medido en spike 31442561/81:
#  13.8K tok/s agg @bs4/rank, 30.9K @bs12/rank, DDP OK con balance_loss).
# El optimizer.pt ZeRO-1 es formato estandar -> resume desde ckpts de
# olas NVIDIA y viceversa (mismo OUT, misma cadena AUTO_RESUBMIT).
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=/beegfs/a474r867/ecoreasoner
VB=/beegfs/a474r867
export ROCM_PATH="/kuhpc/sw/rocm/6.4.3"
export PATH="$VB/venv_rocm/bin:${ROCM_PATH}/bin:$PATH"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# transformers+numpy para el trainer: pydeps_tf451 NO tiene torch
# (verificado) -> no sombrea el torch-ROCm del venv
export PYTHONPATH="/beegfs/a474r867/pydeps_tf451${PYTHONPATH:+:$PYTHONPATH}"
# RCCL en MI210: P2P/XGMI e IB fallan -> transporte SHM (verificado spike)
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=${MASTER_PORT:-29519}
SCRIPTS=$BASE/scripts
DATA_CACHE=${DATA_CACHE:-$BASE/data/train_ids_v5pdb.npz}
OUT=${WAVE_OUT:-$BASE/runs/v5-1b}
WAVE_ID=${WAVE_ID:-v5-1b}
FLAG=$OUT/training_complete.flag
TARGET_STEPS=${TARGET_STEPS:-1300000}
AUTO_RESUBMIT=${AUTO_RESUBMIT:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}
N_EXPERTS=${N_EXPERTS:-16}
NRANKS=${NRANKS:-3}
# batch efectivo = receta NVIDIA: NRANKS*BATCH*ACCUM*768 = 9216 tok/step.
# MI210 64GB -> bs4/rank accum1 EMA-GPU (misma forma que isla NVIDIA >=40GB)
BATCH=${BATCH:-4}; ACCUM=${ACCUM:-1}
EMA_ARGS=${EMA_ARGS:---ema_device cuda}
CKPT_FLAG=${CKPT_FLAG:-}; ZERO_FLAG=${ZERO_FLAG:-}
mkdir -p "$OUT" "$BASE/logs"

echo "[$(date)] V5-1B-MI210 $WAVE_ID ranks=$NRANKS bs=$BATCH acc=$ACCUM (corpus=$DATA_CACHE, target=$TARGET_STEPS)"

FINALIZED=0
finalize() {
    [ "$FINALIZED" = "1" ] && return
    FINALIZED=1
    if [ -f "$FLAG" ]; then echo "complete"; cat "$FLAG"; exit 0; fi
    if [ "$AUTO_RESUBMIT" = "1" ]; then
        echo "relanzando ola $WAVE_ID en mi210..."
        sbatch --export="AUTO_RESUBMIT=1,WAVE_ID=$WAVE_ID,WAVE_OUT=$OUT,TARGET_STEPS=$TARGET_STEPS,DATA_CACHE=$DATA_CACHE,N_EXPERTS=$N_EXPERTS,NRANKS=$NRANKS,BATCH=$BATCH,ACCUM=$ACCUM,EXTRA_ARGS=$EXTRA_ARGS" \
            "$SCRIPTS/v5_1b_mi210.slurm" 2>/dev/null || true
    fi
}
handle_signal() {
    SIGNAL_RECEIVED=1
    echo "=== SIGUSR1 $(date -u) — guardando ola ==="
    if [ -n "${RUN_PID:-}" ] && kill -0 "$RUN_PID"; then
        # torchrun propaga a los hijos? mejor: USR1 al proceso python rank0
        pkill -USR1 -f "train_mdlm_moe_v2.py" 2>/dev/null || true
        set +e; wait "$RUN_PID"; tx=$?; set -e
    fi
    finalize
}
trap handle_signal USR1
trap 'echo SIGTERM; finalize; exit 1' TERM

torchrun --nproc_per_node="$NRANKS" "$SCRIPTS/train_mdlm_moe_v2.py" \
    --data dummy --data_cache "$DATA_CACHE" --output "$OUT" \
    --hidden 768 --layers 12 --heads 12 \
    --n_experts "$N_EXPERTS" --expert_k 1 --ff_mult 4 \
    --seq_len 768 --batch_size "$BATCH" --grad_accum "$ACCUM" \
    --lr 2e-4 --warmup 800 --lr_decay none \
    --mask_type random --mask_schedule uniform \
    --mask_schedule_args '{"b_l":0.30,"b_h":0.99}' \
    --curriculum --candidate_focus 1.0 \
    --ema_decay 0.999 --ckpt_every 500 \
    $CKPT_FLAG $ZERO_FLAG $EMA_ARGS $EXTRA_ARGS \
    --max_steps "$TARGET_STEPS" &
RUN_PID=$!
set +e; wait "$RUN_PID"; TRAIN_X=$?; set -e
echo "=== torchrun exit $TRAIN_X ==="
case "$TRAIN_X" in
    0) echo "COMPLETE" > "$FLAG"; exit 0;;
    42) finalize;;
    *) if [ "${SIGNAL_RECEIVED:-0}" = "1" ]; then finalize; else exit "$TRAIN_X"; fi;;
esac
