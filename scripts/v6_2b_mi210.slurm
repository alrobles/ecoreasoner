#!/bin/bash
#SBATCH --job-name=v6-2b-mi210
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:mi210:3
#SBATCH --constraint="mi210"
#SBATCH --mem=120G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/v6_2b_mi210_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/v6_2b_mi210_%j.err
# V6-2B sobre pool AMD MI210 — brazo de CAPACIDAD de la pregunta abierta:
# "el cuello generativo es capacidad, no receta".
#   MISMO corpus (train_ids_v5pdb.npz), MISMA receta (random+uniform
#   b_l=0.30,b_h=0.99 + curriculum + CF1.0 + EMA 0.999, lr 2e-4 w800),
#   MISMO batch 9216 tok/step, MISMO target 1.3M -> curva comparable
#   punto a punto con v5-1b. UNICO cambio: --layers 12 -> 24
#   (+28% activos ~365M, 2x profundidad, ~2.06B totales).
# Stack: venv_rocm (torch 2.5.1+rocm6.2, SIN apptainer) + torchrun
# intra-nodo + RCCL via SHM (NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1).
# VRAM: bs4/rank ~2x el uso de L12 (~16.4GB -> ~33GB) < 64GB OK.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=/beegfs/a474r867/ecoreasoner
VB=/beegfs/a474r867
export ROCM_PATH="/kuhpc/sw/rocm/6.4.3"
export PATH="$VB/venv_rocm/bin:${ROCM_PATH}/bin:$PATH"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="/beegfs/a474r867/pydeps_tf451${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=${MASTER_PORT:-29519}
SCRIPTS=$BASE/scripts
DATA_CACHE=${DATA_CACHE:-$BASE/data/train_ids_v5pdb.npz}
OUT=${WAVE_OUT:-$BASE/runs/v6-2b}
WAVE_ID=${WAVE_ID:-v6-2b}
FLAG=$OUT/training_complete.flag
TARGET_STEPS=${TARGET_STEPS:-1300000}
AUTO_RESUBMIT=${AUTO_RESUBMIT:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}
N_EXPERTS=${N_EXPERTS:-16}
LAYERS=${LAYERS:-24}
NRANKS=${NRANKS:-3}
# batch efectivo = receta: NRANKS*BATCH*ACCUM*768 = 9216 tok/step
BATCH=${BATCH:-4}; ACCUM=${ACCUM:-1}
EMA_ARGS=${EMA_ARGS:---ema_device cuda}
CKPT_FLAG=${CKPT_FLAG:-}; ZERO_FLAG=${ZERO_FLAG:-}
mkdir -p "$OUT" "$BASE/logs"

echo "[$(date)] V6-2B-MI210 $WAVE_ID L=$LAYERS ranks=$NRANKS bs=$BATCH acc=$ACCUM (corpus=$DATA_CACHE, target=$TARGET_STEPS)"

FINALIZED=0
finalize() {
    [ "$FINALIZED" = "1" ] && return
    FINALIZED=1
    if [ -f "$FLAG" ]; then echo "complete"; cat "$FLAG"; exit 0; fi
    if [ "$AUTO_RESUBMIT" = "1" ]; then
        echo "relanzando ola $WAVE_ID en mi210..."
        for _try in 1 2 3; do
            sout=$(sbatch --export="AUTO_RESUBMIT=1,WAVE_ID=$WAVE_ID,WAVE_OUT=$OUT,TARGET_STEPS=$TARGET_STEPS,DATA_CACHE=$DATA_CACHE,N_EXPERTS=$N_EXPERTS,LAYERS=$LAYERS,NRANKS=$NRANKS,BATCH=$BATCH,ACCUM=$ACCUM,EXTRA_ARGS=$EXTRA_ARGS" \
                "$SCRIPTS/v6_2b_mi210.slurm" 2>&1) && { echo "$sout"; break; }
            echo "SBATCH intento $_try FALLO: $sout"
            sleep 15
        done
    fi
}
handle_signal() {
    SIGNAL_RECEIVED=1
    echo "=== SIGUSR1 $(date -u) — guardando ola ==="
    if [ -n "${RUN_PID:-}" ] && kill -0 "$RUN_PID"; then
        pkill -USR1 -f "train_mdlm_moe_v2.py" 2>/dev/null || true
        set +e; wait "$RUN_PID"; tx=$?; set -e
    fi
    finalize
}
trap handle_signal USR1
trap 'echo SIGTERM; finalize; exit 1' TERM

torchrun --nproc_per_node="$NRANKS" "$SCRIPTS/train_mdlm_moe_v2.py" \
    --data dummy --data_cache "$DATA_CACHE" --output "$OUT" \
    --hidden 768 --layers "$LAYERS" --heads 12 \
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
