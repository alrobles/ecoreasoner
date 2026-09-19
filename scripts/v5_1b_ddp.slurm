#!/bin/bash
#SBATCH --job-name=v5-1b
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --constraint="q6000|a100|l40|a40|q8000|pro6000"
#SBATCH --mem=170G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/v5_1b_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/v5_1b_%j.err
# V5-1B: pretrain DESDE CERO del MdLMMoE ~1.13B (768h/12L/16 expertos k=1,
# ~285M activos) sobre train_ids_v5pdb.npz (11.96B tok, 17.4M docs).
# Receta = la campeona del GA: random+uniform b_l=0.30,b_h=0.99 +
# curriculum + CF1.0 + EMA 0.999. Olas: SIGUSR1@300 + AUTO_RESUBMIT
# (patron mdlm_retrain_v4: cada ola resume el ultimo checkpoint-gN).
#
# FLOTA FLEXIBLE: constraint+gres=gpu:3 aterriza en CUALQUIER isla NVIDIA
# >=3 GPUs (q6000x3/4, a100x3, l40x4, a40x4; excluye mi210=AMD y v100<3GPU).
# Auto-adapta al VRAM: >=40GB -> bs4 accum1 (sin ckpt/zero1);
# <40GB (q6000) -> bs1 accum4 + grad_ckpt + --zero1 (opt shardado /N).
# El optimizer.pt consolidado ZeRO es formato estandar -> las olas pueden
# migrar entre configs sin romper resume.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
SCRIPTS=$BASE/scripts
DATA_CACHE=${DATA_CACHE:-$BASE/data/train_ids_v5pdb.npz}
OUT=${WAVE_OUT:-$BASE/runs/v5-1b}
WAVE_ID=${WAVE_ID:-v5-1b}
FLAG=$OUT/training_complete.flag
TARGET_STEPS=${TARGET_STEPS:-1300000}
AUTO_RESUBMIT=${AUTO_RESUBMIT:-1}
N_EXPERTS=${N_EXPERTS:-16}
NTASKS=${SLURM_NTASKS:-3}
mkdir -p "$OUT" "$BASE/logs"

# --- auto-adaptacion por VRAM (env override gana) ---
# NB: sin `head` — con pipefail, head cierra el pipe y nvidia-smi muere por
# SIGPIPE (141) de forma aleatoria. sed/awk leen todo el input.
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | sed -n '1p' | tr -dc '0-9')
VRAM_MIB=${VRAM_MIB:-0}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n '1p' || echo "?")
if [ "${VRAM_MIB:-0}" -ge 40000 ]; then
    BATCH=${BATCH:-4}; ACCUM=${ACCUM:-1}; CKPT_FLAG=${CKPT_FLAG:-}; ZERO_FLAG=${ZERO_FLAG:-}
    EMA_ARGS=${EMA_ARGS:---ema_device cuda}   # EMA en GPU: en CPU cuesta ~25s/step (1.1B params)
else
    BATCH=${BATCH:-1}; ACCUM=${ACCUM:-4}; CKPT_FLAG=${CKPT_FLAG:---grad_ckpt}; ZERO_FLAG=${ZERO_FLAG:---zero1}
    EMA_ARGS=${EMA_ARGS:---ema_device cpu --ema_every 8}  # q6000 no cabe EMA-GPU; amortizar 8x
fi
# batch efectivo constante: NTASKS*BATCH*ACCUM*768 tok/step en todas las islas

echo "[$(date)] V5-1B $WAVE_ID DDP=$NTASKS gpu=$GPU_NAME vram=${VRAM_MIB}MiB bs=$BATCH acc=$ACCUM (corpus=$DATA_CACHE, target=$TARGET_STEPS)"

FINALIZED=0
finalize() {
    [ "$FINALIZED" = "1" ] && return
    FINALIZED=1
    if [ -f "$FLAG" ]; then echo "complete"; cat "$FLAG"; exit 0; fi
    if [ "$AUTO_RESUBMIT" = "1" ]; then
        echo "relanzando ola $WAVE_ID..."
        sbatch --export="AUTO_RESUBMIT=1,WAVE_ID=$WAVE_ID,WAVE_OUT=$OUT,TARGET_STEPS=$TARGET_STEPS,DATA_CACHE=$DATA_CACHE,N_EXPERTS=$N_EXPERTS" \
            "$SCRIPTS/v5_1b_ddp.slurm" 2>/dev/null || true
    fi
}
handle_signal() {
    echo "=== SIGUSR1 $(date -u) — guardando ola ==="
    if [ -n "${SRUN_PID:-}" ] && kill -0 "$SRUN_PID"; then
        kill -USR1 "$SRUN_PID" 2>/dev/null || true
        set +e; wait "$SRUN_PID"; tx=$?; set -e
    fi
    finalize
}
trap handle_signal USR1
trap 'echo SIGTERM; exit 1' TERM

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29517
export OMPI_MCA_btl=^openib

# srun propaga MASTER_* a todos los ranks (critico multi-nodo; intra-nodo
# tambien funciona). v2 resuelve master via SLURM_JOB_NODELIST si falta.
srun --export=ALL -n "$NTASKS" /usr/bin/apptainer exec --nv \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$SIF" python3 "$SCRIPTS/train_mdlm_moe_v2.py" \
        --data dummy --data_cache "$DATA_CACHE" --output "$OUT" \
        --hidden 768 --layers 12 --heads 12 \
        --n_experts "$N_EXPERTS" --expert_k 1 --ff_mult 4 \
        --seq_len 768 --batch_size "$BATCH" --grad_accum "$ACCUM" \
        --lr 2e-4 --warmup 800 --lr_decay none \
        --mask_type random --mask_schedule uniform \
        --mask_schedule_args '{"b_l":0.30,"b_h":0.99}' \
        --curriculum --candidate_focus 1.0 \
        --ema_decay 0.999 --ckpt_every 500 \
        $CKPT_FLAG $ZERO_FLAG $EMA_ARGS \
        --max_steps "$TARGET_STEPS" &
SRUN_PID=$!
set +e; wait "$SRUN_PID"; TRAIN_X=$?; set -e
echo "=== srun exit $TRAIN_X ==="
case "$TRAIN_X" in
    0) echo "COMPLETE" > "$FLAG"; exit 0;;
    42) finalize;;
    *) if [ "${SIGNAL_RECEIVED:-0}" = "1" ]; then finalize; else exit "$TRAIN_X"; fi;;
esac
