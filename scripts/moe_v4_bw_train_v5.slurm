#!/usr/bin/env bash
#SBATCH --job-name=moe-v4-bw-v5
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=80G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_v4_bw_v5_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_v4_bw_v5_%j.err

# =============================================================================
# EcoReasoner dLLM MoE v4 — ENTRENAMIENTO en Blackwell (2x RTX PRO 6000, sm_120)
# con CORPUS v5 LIMPIO (train_ids_v5_clean.npy). Trainer VANILLA
# (train_mdlm_moe.py), el UNICO probado estable en DDP world=2 en estas GPUs
# (bw0 corrio horas; el lsgd train_mdlm_moe_lsgd.py da IMA en DDP world>=2,
# falla en top-1/top-2/clip/no_sync por igual - 2026-08-29).
#
# - 1 contenedor por nodo (ntasks=1) + torchrun --nproc_per_node=2 DENTRO.
# - MoE 4 expertos top-1, DDP world=2 INTRA-NODO (batch 4/rank). NO multi-nodo
#   (all-reduce por experto con batch pequeno -> hang, leccion 2026-08-27).
# - max_steps 6000 con encadenamiento por olas (SIGUSR1@300 + AUTO_RESUBMIT).
# - Fix resume 2026-08-29: fallback a ultimo checkpoint integro + retener 2.
# =============================================================================
set -euo pipefail

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
WAVE_ID="${WAVE_ID:-bw1v5}"
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
            "$BASE/scripts/moe_v4_bw_train_v5.slurm" 2>/dev/null || \
        sbatch "$BASE/scripts/moe_v4_bw_train_v5.slurm"
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
apptainer exec --nv --overlay "$OVL" "$SIF" "$VENV/bin/python" -c "
import torch
print('GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(' ', torch.cuda.get_device_name(i))
assert torch.cuda.device_count() >= 2
" || { echo 'GPU detect FAIL'; exit 1; }

# ---- Lanzar entrenamiento MoE v4 (2x pro6000, torchrun intra-nodo) ---------
echo "=== Lanzando MoE v4 Blackwell v5 ($WAVE_ID -> $OUT, steps $TARGET_STEPS) $(date -u) ==="
apptainer exec --nv --overlay "$OVL" "$SIF" \
    "$VENV/bin/python" -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29512 \
    "$BASE/scripts/train_mdlm_moe.py" \
    --data_cache "$BASE/data/train_ids_v5_clean.npy" \
    --data "$BASE/data/train_corpus_v5.jsonl" \
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
