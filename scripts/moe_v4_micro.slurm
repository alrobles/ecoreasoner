#!/usr/bin/env bash
#SBATCH --job-name=f0-micro
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=80G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/f0_micro_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/f0_micro_%j.err
# =============================================================================
# F0 micro-run (Fase 3 -new): entrena un micro-modelo DENSE 50-100M desde cero
# con UN objetivo de masking sobre UN corpus, y al completar lo evalúa con
# suite_smoke (discriminación inferencial pairwise + completación + fluidez)
# y ensambla report.json + index.jsonl. 1 GPU pro6000 (Blackwell).
#
# Env (pasadas con --export):
#   TAG         tag del run (f0-random-prosa, f0-spanhi-esqueleto, ...)
#   MASK_TYPE   random | span
#   MASK_P      ratio de máscara (0.15 | 1.2 para span-alto)
#   SPAN_LEN    longitud de span (solo span; default 64)
#   DATA_CACHE  npy de tokens (prosa: train_ids_v7_clean.npy | esqueleto: train_ids_skeleton.npy)
#   TARGET_STEPS  steps absolutos (default 10000)
#   PAIRS       jsonl de pares para la eval de discriminación
#   CONFIG      run.yaml del harness (para report.py)
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
RUNS=$BASE/runs
TAG="${TAG:?TAG requerido}"
MASK_TYPE="${MASK_TYPE:?MASK_TYPE requerido}"
MASK_P="${MASK_P:?MASK_P requerido}"
SPAN_LEN="${SPAN_LEN:-64}"
DATA_CACHE="${DATA_CACHE:?DATA_CACHE requerido}"
TARGET_STEPS="${TARGET_STEPS:-10000}"
PAIRS="${PAIRS:-$RUNS/pairs.jsonl}"
CONFIG="${CONFIG:?CONFIG requerido}"
OUT=$RUNS/$TAG
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
mkdir -p "$OUT"
SIGNAL_RECEIVED=0
FINALIZED=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_P2P_DISABLE=1

finalize() {
    if [ "$FINALIZED" = "1" ]; then echo "=== finalize ya ejecutado, skip ==="; return; fi
    FINALIZED=1
    echo "=== finalize $(date -u) ==="
    if [ -f "$FLAG" ]; then
        echo "training complete; flag=$FLAG"; cat "$FLAG"; exit 0
    fi
    if [ "$AUTO_RESUBMIT" = "1" ]; then
        echo "resubmiting ola ($TAG) target=$TARGET_STEPS..."
        sbatch --export="TAG=$TAG,MASK_TYPE=$MASK_TYPE,MASK_P=$MASK_P,SPAN_LEN=$SPAN_LEN,DATA_CACHE=$DATA_CACHE,TARGET_STEPS=$TARGET_STEPS,PAIRS=$PAIRS,CONFIG=$CONFIG,AUTO_RESUBMIT=1" \
            "$BASE/scripts/moe_v4_micro.slurm" 2>/dev/null || \
        sbatch "$BASE/scripts/moe_v4_micro.slurm"
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

apptainer exec --nv --overlay "$OVL":ro "$SIF" "$VENV/bin/python" -c "
import torch
print('GPUs:', torch.cuda.device_count())
assert torch.cuda.device_count() >= 1, 'sin GPU'
" || { echo 'GPU detect FAIL'; exit 1; }

echo "=== Lanzando micro-run $TAG (mask=$MASK_TYPE p=$MASK_P span=$SPAN_LEN, steps=$TARGET_STEPS) $(date -u) ==="
apptainer exec --nv --overlay "$OVL":ro "$SIF" \
    "$VENV/bin/python" "$BASE/scripts/train_mdlm_moe.py" \
    --data_cache "$DATA_CACHE" \
    --data "$DATA_CACHE" \
    --vocab 126080 --hidden 512 --layers 8 --heads 8 --ff_mult 4 \
    --n_experts 1 --expert_k 1 --seq_len 768 \
    --batch_size 8 --grad_accum 2 --mask_p "$MASK_P" \
    --mask_type "$MASK_TYPE" --span_len "$SPAN_LEN" \
    --lr 2e-4 --warmup 200 --max_steps "$TARGET_STEPS" \
    --output "$OUT" &
TRAIN_PID=$!
set +e; wait "$TRAIN_PID"; TRAIN_X=$?; set -e
echo "=== train exit $TRAIN_X ==="
case "$TRAIN_X" in
  0) echo "COMPLETE" > "$FLAG";;
  42) finalize;;
  *) if [ "$SIGNAL_RECEIVED" = "1" ]; then finalize; else
        echo "fail $TRAIN_X"; exit "$TRAIN_X"; fi;;
esac

# ---- eval: checkpoint más reciente + suite_smoke + report ----
CKPT_DIR=$(ls -d "$OUT"/checkpoint-g* 2>/dev/null | sort -t g -k2 -n | tail -1 || true)
if [ -z "$CKPT_DIR" ]; then
    echo "[eval] sin checkpoints en $OUT"; exit 1
fi
CKPT="$CKPT_DIR/model.pt"
echo "=== eval $CKPT con $PAIRS ==="
apptainer exec --nv --overlay "$OVL":ro "$SIF" "$VENV/bin/python" \
    "$BASE/harness/suite_smoke.py" \
    --ckpt "$CKPT" --config "$CONFIG" --pairs "$PAIRS" \
    --out "$OUT/report.suite.json" --device cuda || echo "[eval] suite FAILED"
apptainer exec --nv --overlay "$OVL":ro "$SIF" "$VENV/bin/python" \
    "$BASE/harness/report.py" \
    --config "$CONFIG" --suite "$OUT/report.suite.json" \
    --out "$OUT/report.json" --job-id "$SLURM_JOB_ID" || echo "[eval] report FAILED"
echo "=== micro-run $TAG DONE $(date -u) ==="