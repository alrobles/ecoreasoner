#!/usr/bin/env bash
#SBATCH --job-name=moe-v4-bw-v7
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=80G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_v4_bw_v7_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_v4_bw_v7_%j.err

# =============================================================================
# EcoReasoner dLLM MoE v4 — ATAQUE DE RAIZ al smoke L1 (bw3, 2026-08-30).
# Continuacion WARM-START desde bw1v5/checkpoint-g6000 (loss 5.35, 37M tok vistos)
# sobre el CORPUS CURADO v7 (train_ids_v7_clean.npy, 783.2M tok, 1.46M docs).
# Trainer VANILLA (train_mdlm_moe.py) — el unico probado estable en DDP world=2
# Blackwell (lsgd = IMA en DDP; NCCL 2.26.2 buggy con sm_120).
#
# Dimensionado (2026-08-30): grad_accum 4 -> optimizer-step = 24,576 tok
# (2 ranks x 4 batch x 4 accum x 768 seq). 1 epoch v7 = 31,865 steps.
# TARGET_STEPS = 6000 + 4*31865 = 133,459 -> 133,500 (4 epochs, ~6.3 dias).
# Revision a 2 epochs (~69,700): si loss > 4.2 y sin texto estructurado -> parar.
# Olas SIGUSR1@300 + AUTO_RESUBMIT (patron bw1v5, resume tolerante validado).
# =============================================================================
set -euo pipefail

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
WAVE_ID="${WAVE_ID:-bw3}"
OUT="${WAVE_OUT:-$BASE/outputs/$WAVE_ID}"
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
TARGET_STEPS="${TARGET_STEPS:-133500}"
mkdir -p "$OUT"

# ---- Slurm trap (ola: SIGUSR1 300s antes del fin de walltime) --------------
SIGNAL_RECEIVED=0
FINALIZED=0

# NCCL en Blackwell 2-GPU (2026-09-01): con DDP REAL activo (fix de env vars),
# NCCL 2.26.2 daba "illegal memor access" en el all-reduce. se actualizó el
# .s de nvidia-nccl a 2.26.5 en el overlay; ademas workarunds documentados:
# CUMEM_HOST_ENABLE=0 (desbloquea path IMA/OGM) + P2P_DISABLE=1 (evita hang
# P2P en Blackwell; con 2 GPUs intra-nodo la perdida de rendimiento es minima).
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_P2P_DISABLE=1
finalize() {
    if [ "$FINALIZED" = "1" ]; then
        echo "=== finalize ya ejecutado, skip (evita doble resubmit) ==="
        return
    fi
    FINALIZED=1
    echo "=== finalize $(date -u) ==="
    if [ -f "$FLAG" ]; then
        echo "training complete; flag=$FLAG"; cat "$FLAG"; exit 0
    fi
    if [ "$AUTO_RESUBMIT" = "1" ]; then
        echo "resubmiting ola ($WAVE_ID) target=$TARGET_STEPS..."
        sbatch --export="AUTO_RESUBMIT=1,TARGET_STEPS=$TARGET_STEPS,WAVE_ID=$WAVE_ID,WAVE_OUT=$OUT" \
            "$BASE/scripts/moe_v4_bw_v7.slurm" 2>/dev/null || \
        sbatch "$BASE/scripts/moe_v4_bw_v7.slurm"
        echo "ola siguiente enviada."
    else
        echo "auto-resubmit disabled"
    fi
    exit 42
}
handle_signal() {
    echo "=== SIGUSR1 recibido $(date -u) ==="
    SIGNAL_RECEIVED=1
    pkill -USR1 -f "train_mdlm_moe.py" 2>/dev/null || true
    if [ -n "${TRAIN_PID:-}" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -USR1 "$TRAIN_PID" 2>/dev/null || true
    fi
    set +e; wait "$TRAIN_PID" 2>/dev/null; train_x=$?; set -e
    echo "trainer cerrado tras señal (exit $train_x)"
    finalize
}
trap handle_signal USR1
trap 'echo SIGTERM; exit 1' TERM

# ---- Verificación GPU -----------------------------------------------------
apptainer exec --nv --overlay "$OVL":ro "$SIF" "$VENV/bin/python" -c "
import torch
print('GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(' ', torch.cuda.get_device_name(i))
assert torch.cuda.device_count() >= 2
" || { echo 'GPU detect FAIL'; exit 1; }

# ---- Lanzar entrenamiento MoE v4 (2x pro6000, torchrun intra-nodo) ---------
echo "=== Lanzando MoE v4 bw3 ($WAVE_ID -> $OUT, steps $TARGET_STEPS) $(date -u) ==="
apptainer exec --nv --overlay "$OVL":ro "$SIF" \
    "$VENV/bin/python" -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29512 \
    "$BASE/scripts/train_mdlm_moe.py" \
    --data_cache "$BASE/data/train_ids_v7_clean.npy" \
    --data "$BASE/data/curate/train_corpus_v7_curated.jsonl" \
    --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
    --n_experts 4 --expert_k 1 --seq_len 768 \
    --batch_size 4 --grad_accum 4 --mask_p 0.15 \
    --lr 1.5e-4 --warmup 1000 --max_steps "$TARGET_STEPS" \
    --output "$OUT" &
TRAIN_PID=$!
set +e; wait "$TRAIN_PID"; TRAIN_X=$?; set -e
echo "=== train exit $TRAIN_X ==="
case "$TRAIN_X" in
  0) echo "COMPLETE" > "$FLAG"; exit 0;;
  42) finalize;;
  *) if [ "$SIGNAL_RECEIVED" = "1" ]; then finalize; else
        echo "fail $TRAIN_X"; exit "$TRAIN_X"; fi;;
esac