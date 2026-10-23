#!/bin/bash
#SBATCH --job-name=v6-2b-cnr
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|l40|a40|q8000|pro6000"
#SBATCH --prefer="pro6000|a100|l40|a40|q8000"
#SBATCH --mem=30G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/v6_2b_cnr_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/v6_2b_cnr_%j.err
# V6-2B-NV: brazo de CAPACIDAD (layers 12->24, ~365M activos, 2.06B total)
# portado a pool NVIDIA 10-22 tras 6 dias de crash-loop RCCL/SHM en MI210
# (deadlock aleatorio de colectivas — py-spy: hang en DDP init/allreduce).
# MISMO corpus/receta/batch 9216 tok/step que v5-1b y que el intento MI210;
# resume runs/v6-2b/checkpoint-g12501 (formato ckpt identico cross-stack).
# q6000 (24GB) queda fuera: costos fijos ~41GB no caben.
# Tiers VRAM: >=60G -> bs4 acc1 EMA-GPU; 40-60G -> bs2 acc2 +zero1
# +grad_ckpt +EMA-GPU (~42GB); <40G -> bs1 acc4 +ckpt +zero1 +EMA-CPU.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
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
NTASKS=${SLURM_NTASKS:-3}
SHAPE_N=${SHAPE_N:-$NTASKS}
SHAPE_G=${SHAPE_G:-gpu:$NTASKS}
mkdir -p "$OUT" "$BASE/logs"

# --- auto-adaptacion por VRAM (env override gana) ---
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -dc '0-9')
VRAM_MIB=${VRAM_MIB:-0}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n '1p' || echo "?")
# batch efectivo constante NTASKS*BATCH*ACCUM*768 = 9216 tok/step
if [ "${VRAM_MIB:-0}" -ge 60000 ]; then
    BATCH=${BATCH:-4}; ACCUM=${ACCUM:-1}; CKPT_FLAG=${CKPT_FLAG:-}; ZERO_FLAG=${ZERO_FLAG:-}
    EMA_ARGS=${EMA_ARGS:---ema_device cuda}
elif [ "${VRAM_MIB:-0}" -ge 40000 ]; then
    # islas ~44-48GB (a40/l40/q8000/a100-40): bs2+ckpt+zero1+EMA-CPU
    # (EMA-GPU OOMaba: 43.9/44.4GiB en cross_entropy — medido 10-22)
    BATCH=${BATCH:-2}; ACCUM=${ACCUM:-2}; CKPT_FLAG=${CKPT_FLAG:---grad_ckpt}; ZERO_FLAG=${ZERO_FLAG:---zero1}
    EMA_ARGS=${EMA_ARGS:---ema_device cpu --ema_every 8}
else
    # defensivo (no deberia ocurrir con el constraint actual)
    BATCH=${BATCH:-1}; ACCUM=${ACCUM:-4}; CKPT_FLAG=${CKPT_FLAG:---grad_ckpt}; ZERO_FLAG=${ZERO_FLAG:---zero1}
    EMA_ARGS=${EMA_ARGS:---ema_device cpu --ema_every 8}
fi

echo "[$(date)] V6-2B-NV $WAVE_ID L=$LAYERS DDP=$NTASKS gpu=$GPU_NAME vram=${VRAM_MIB}MiB bs=$BATCH acc=$ACCUM (corpus=$DATA_CACHE, target=$TARGET_STEPS)"

FINALIZED=0
finalize() {
    [ "$FINALIZED" = "1" ] && return
    FINALIZED=1
    if [ -f "$FLAG" ]; then echo "complete"; cat "$FLAG"; exit 0; fi
    if [ "$AUTO_RESUBMIT" = "1" ]; then
        local sn=$SHAPE_N sg=$SHAPE_G
        if [ -f "$OUT/.wd_shape" ]; then
            sn=$(cat "$OUT/.wd_shape" 2>/dev/null); rm -f "$OUT/.wd_shape"
            sg="gpu:$sn"; echo "watchdog pidio cambio de forma -> ntasks=$sn"
        fi
        echo "relanzando ola $WAVE_ID (ntasks=$sn gres=$sg)..."
        for _try in 1 2 3; do
            sout=$(sbatch --ntasks="$sn" --ntasks-per-node="$sn" --gres="$sg" \
                --export="AUTO_RESUBMIT=1,WAVE_ID=$WAVE_ID,WAVE_OUT=$OUT,TARGET_STEPS=$TARGET_STEPS,DATA_CACHE=$DATA_CACHE,N_EXPERTS=$N_EXPERTS,LAYERS=$LAYERS,SHAPE_N=$sn,SHAPE_G=$sg,EXTRA_ARGS=$EXTRA_ARGS" \
                "$SCRIPTS/v6_2b_nv.slurm" 2>&1) && { echo "$sout"; break; }
            echo "SBATCH intento $_try FALLO: $sout"
            sleep 15
        done
    fi
}
handle_signal() {
    SIGNAL_RECEIVED=1
    echo "=== SIGUSR1 $(date -u) — guardando ola ==="
    if [ -n "${SRUN_PID:-}" ] && kill -0 "$SRUN_PID"; then
        kill -USR1 "$SRUN_PID" 2>/dev/null || true
        set +e; wait "$SRUN_PID"; tx=$?; set -e
    fi
    finalize
}
trap handle_signal USR1
trap 'echo SIGTERM; finalize; exit 1' TERM

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29519
export OMPI_MCA_btl=^openib

srun --export=ALL -n "$NTASKS" /usr/bin/apptainer exec --nv \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$SIF" python3 "$SCRIPTS/train_mdlm_moe_v2.py" \
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
SRUN_PID=$!
set +e; wait "$SRUN_PID"; TRAIN_X=$?; set -e
echo "=== srun exit $TRAIN_X ==="
case "$TRAIN_X" in
    0) echo "COMPLETE" > "$FLAG"; exit 0;;
    42) finalize;;
    *) if [ "${SIGNAL_RECEIVED:-0}" = "1" ]; then finalize; else exit "$TRAIN_X"; fi;;
esac
