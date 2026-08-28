#!/usr/bin/env bash
#SBATCH --job-name=moe-v4-bw
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=80G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_v4_bw_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_v4_bw_%j.err

# =============================================================================
# EcoReasoner dLLM MoE v4 — ENTRENAMIENTO REAL en Blackwell (2x RTX PRO 6000,
# sm_120). Escala del smoke validado (27477584): mismo modelo y config, pero
# max_steps 6000 con encadenamiento por olas (SIGUSR1@300 + AUTO_RESUBMIT).
#
# Patrón de lanzamiento validado 2026-08-28:
#   - 1 contenedor por nodo (ntasks=1) + torchrun --nproc_per_node=2 DENTRO,
#     para NO montar el mismo overlay ext3 en paralelo (lock -> exit 255).
#   - torch 2.7.1+cu128 en el venv /bb/bwvenv dentro de blackwell-torch.overlay.
#   - MoE 4 expertos top-1, DDP world=2 INTRA-NODO (batch 4/rank). NO multi-nodo:
#     all-reduce por experto con batch pequeño -> hang (lección 2026-08-27).
#   - El trainer controla checkpoint/exit 42; el bash relanza la siguiente ola.
# =============================================================================
set -euo pipefail

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
WAVE_ID="${WAVE_ID:-bw0}"
OUT="${WAVE_OUT:-$BASE/outputs/$WAVE_ID}"
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
TARGET_STEPS="${TARGET_STEPS:-6000}"
mkdir -p "$OUT"

# ---- Slurm trap (ola: SIGUSR1 300s antes del fin de walltime) --------------
SIGNAL_RECEIVED=0
FINALIZED=0
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
            "$BASE/scripts/moe_v4_bw_train.slurm" 2>/dev/null || \
        sbatch "$BASE/scripts/moe_v4_bw_train.slurm"
        echo "ola siguiente enviada."
    else
        echo "auto-resubmit disabled"
    fi
    exit 42
}
handle_signal() {
    echo "=== SIGUSR1 recibido $(date -u) ==="
    SIGNAL_RECEIVED=1
    # 1) Señal directa a los workers python (guarantiza checkpoint+exit 42)
    pkill -USR1 -f "train_mdlm_moe.py" 2>/dev/null || true
    # 2) También al agente torchrun por si propaga él
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
apptainer exec --nv --overlay "$OVL" "$SIF" "$VENV/bin/python" -c "
import torch
print('GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(' ', torch.cuda.get_device_name(i))
assert torch.cuda.device_count() >= 2
" || { echo 'GPU detect FAIL'; exit 1; }

# ---- Lanzar entrenamiento MoE v4 (2x pro6000, torchrun intra-nodo) ---------
echo "=== Lanzando MoE v4 Blackwell ($WAVE_ID -> $OUT, steps $TARGET_STEPS) $(date -u) ==="
apptainer exec --nv --overlay "$OVL" "$SIF" \
    "$VENV/bin/python" -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29512 \
    "$BASE/scripts/train_mdlm_moe.py" \
    --data_cache "$BASE/data/train_ids_v4.npy" \
    --data "$BASE/data/train_corpus_v4.jsonl" \
    --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
    --n_experts 4 --expert_k 1 --seq_len 768 \
    --batch_size 4 --grad_accum 4 --mask_p 0.15 \
    --lr 2e-4 --warmup 800 --max_steps "$TARGET_STEPS" \
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