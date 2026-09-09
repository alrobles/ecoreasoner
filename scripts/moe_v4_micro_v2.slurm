#!/usr/bin/env bash
#SBATCH --job-name=f0-micro-v2
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=80G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/f0_micro_v2_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/f0_micro_v2_%j.err
# =============================================================================
# F0 micro-run v2 (Fase 3 -new): entrena con train_mdlm_moe_v2.py.
# Receta v2: packing EOS (data_cache .npz), mask_schedule uniform, whole_stage,
# cosine/WSD LR, grad_clip, EMA, weight_tying, RoPE opcional.
#
# Env:
#   TAG            tag del run
#   DATA_CACHE     .npz o .npy de tokens
#   CONFIG         run.yaml v2
#   TARGET_STEPS   default 10000
#   PAIRS          jsonl de pares
#   MASK_TYPE      random | span
#   MASK_SCHEDULE  fixed | uniform | cosine
#   MASK_SCHEDULE_ARGS  JSON, default '{"b_l":0.05,"b_h":0.95}'
#   SPAN_LEN       default 64
#   WHOLE_STAGE    0 | 1
#   LR             default 2e-4
#   LR_DECAY       none | cosine | wsd
#   LR_MIN_RATIO   default 0.1
#   WARMUP         default 200
#   GRAD_CLIP      default 1.0
#   EMA_DECAY      default 0.0
#   WEIGHT_TYING   0 | 1
#   USE_ROPE       0 | 1
#   BATCH_SIZE     default 8
#   GRAD_ACCUM     default 2
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
RUNS=$BASE/runs
TAG="${TAG:?TAG requerido}"
DATA_CACHE="${DATA_CACHE:?DATA_CACHE requerido}"
CONFIG="${CONFIG:?CONFIG requerido}"
TARGET_STEPS="${TARGET_STEPS:-10000}"
PAIRS_DIR="${PAIRS_DIR:-$RUNS/pairs_hard_v2}"
PAIRS="${PAIRS:-$PAIRS_DIR/pairs_L0.jsonl}"
MASK_TYPE="${MASK_TYPE:-span}"
MASK_SCHEDULE="${MASK_SCHEDULE:-uniform}"
if [ -z "${MASK_SCHEDULE_ARGS:-}" ]; then
    MASK_SCHEDULE_ARGS='{"b_l":0.05,"b_h":0.95}'
fi
SPAN_LEN="${SPAN_LEN:-64}"
WHOLE_STAGE="${WHOLE_STAGE:-1}"
LR="${LR:-2e-4}"
LR_DECAY="${LR_DECAY:-cosine}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
WARMUP="${WARMUP:-200}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EMA_DECAY="${EMA_DECAY:-0.0}"
WEIGHT_TYING="${WEIGHT_TYING:-0}"
USE_ROPE="${USE_ROPE:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
OUT=$RUNS/$TAG
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
mkdir -p "$OUT"
SIGNAL_RECEIVED=0
FINALIZED=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_P2P_DISABLE=1

# Funcion auxiliar para argumentos booleanos
flag() { [ "$1" = "1" ] && echo "$2" || echo ""; }

finalize() {
    if [ "$FINALIZED" = "1" ]; then echo "=== finalize ya ejecutado, skip ==="; return; fi
    FINALIZED=1
    echo "=== finalize $(date -u) ==="
    if [ -f "$FLAG" ]; then
        echo "training complete; flag=$FLAG"; cat "$FLAG"; exit 0
    fi
    if [ "$AUTO_RESUBMIT" = "1" ]; then
        echo "resubmiting ola ($TAG) target=$TARGET_STEPS..."
        sbatch --export="TAG=$TAG;DATA_CACHE=$DATA_CACHE;CONFIG=$CONFIG;TARGET_STEPS=$TARGET_STEPS;PAIRS_DIR=$PAIRS_DIR;PAIRS=$PAIRS;MASK_TYPE=$MASK_TYPE;MASK_SCHEDULE=$MASK_SCHEDULE;SPAN_LEN=$SPAN_LEN;WHOLE_STAGE=$WHOLE_STAGE;LR=$LR;LR_DECAY=$LR_DECAY;LR_MIN_RATIO=$LR_MIN_RATIO;WARMUP=$WARMUP;GRAD_CLIP=$GRAD_CLIP;EMA_DECAY=$EMA_DECAY;WEIGHT_TYING=$WEIGHT_TYING;USE_ROPE=$USE_ROPE;BATCH_SIZE=$BATCH_SIZE;GRAD_ACCUM=$GRAD_ACCUM;AUTO_RESUBMIT=1" \
            "$BASE/scripts/moe_v4_micro_v2.slurm"
        echo "ola siguiente enviada."
    else
        echo "auto-resubmit disabled"
    fi
    exit 42
}
handle_signal() {
    echo "=== SIGUSR1 recibido $(date -u) ==="
    SIGNAL_RECEIVED=1
    pkill -USR1 -f "train_mdlm_moe_v2.py" 2>/dev/null || true
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

echo "=== Lanzando micro-run v2 $TAG (mask=$MASK_TYPE schedule=$MASK_SCHEDULE steps=$TARGET_STEPS) $(date -u) ==="

apptainer exec --nv --overlay "$OVL":ro "$SIF" \
    "$VENV/bin/python" "$BASE/scripts/train_mdlm_moe_v2.py" \
    --data_cache "$DATA_CACHE" \
    --vocab 126080 --hidden 512 --layers 8 --heads 8 --ff_mult 4 \
    --n_experts 1 --expert_k 1 --seq_len 768 \
    --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" \
    --mask_type "$MASK_TYPE" --mask_schedule "$MASK_SCHEDULE" \
    --mask_schedule_args "$MASK_SCHEDULE_ARGS" \
    --span_len "$SPAN_LEN" $(flag "$WHOLE_STAGE" --whole_stage) \
    --lr "$LR" --lr_decay "$LR_DECAY" --lr_min_ratio "$LR_MIN_RATIO" \
    --warmup "$WARMUP" --max_steps "$TARGET_STEPS" \
    --grad_clip "$GRAD_CLIP" --ema_decay "$EMA_DECAY" \
    $(flag "$WEIGHT_TYING" --weight_tying) \
    $(flag "$USE_ROPE" --use_rope) \
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

# ---- eval: checkpoint mas reciente + suite_smoke_v2 L0-L3 + verdict ----
CKPT_DIR=$(ls -d "$OUT"/checkpoint-g* 2>/dev/null | sort -V | tail -1 || true)
if [ -z "$CKPT_DIR" ]; then
    echo "[eval] sin checkpoints en $OUT"; exit 1
fi
CKPT="$CKPT_DIR/model.pt"
echo "=== eval $CKPT con bateria $PAIRS_DIR ==="
ROPE_FLAG=$(flag "$USE_ROPE" --use-rope)
TIE_FLAG=$(flag "$WEIGHT_TYING" --weight-tying)

apptainer exec --nv --overlay "$OVL":ro "$SIF" "$VENV/bin/python" \
    "$BASE/harness/suite_smoke_v2.py" \
    --ckpt "$CKPT" --config "$CONFIG" --pairs-dir "$PAIRS_DIR" \
    --out-dir "$OUT/battery" --device cuda \
    $ROPE_FLAG $TIE_FLAG || echo "[eval] suite FAILED"

apptainer exec --nv --overlay "$OVL":ro "$SIF" "$VENV/bin/python" \
    "$BASE/scripts/verdict.py" --battery "$OUT/battery" \
    --out "$OUT/battery_verdict.json" || echo "[eval] verdict FAILED"

apptainer exec --nv --overlay "$OVL":ro "$SIF" "$VENV/bin/python" \
    "$BASE/harness/report.py" \
    --config "$CONFIG" --suite "$OUT/battery/battery.json" \
    --out "$OUT/report.json" --job-id "$SLURM_JOB_ID" \
    --index "$BASE/runs/index.jsonl" || echo "[eval] report FAILED"
echo "=== micro-run v2 $TAG DONE $(date -u) ==="
