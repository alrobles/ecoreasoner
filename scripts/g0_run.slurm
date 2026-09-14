#!/usr/bin/env bash
#SBATCH --job-name=g0
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/g0_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/g0_%j.err
# EvoG0 runner generico: receta baseline = B4 (span64 + curriculum + whole_stage
# + role_mask + candidate_focus 0.30, ~90M, 10K steps). Todos los knobs se
# sobreescriben por entorno: sbatch --export=ALL,TAG=g0-x,VAR=... g0_run.slurm
# Auto-detecta Blackwell (sm_120): overlay+venv solo ahi; otros GPUs usan el
# torch base del SIF. Battery logicdiff dense+base + verdicts al COMPLETE.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
VENV=/bb/bwvenv
RUNS=$BASE/runs

TAG=${TAG:-g0-ctrl}
DATA_CACHE=${DATA_CACHE:-$BASE/data/train_ids_b4.npz}
CONFIG=${CONFIG:-$BASE/harness/configs/f0-span-v3-role.yaml}
TARGET_STEPS=${TARGET_STEPS:-10000}
PAIRS_DIR=${PAIRS_DIR:-$RUNS/pairs_hard_v3_eval}
EVAL_CFG=${EVAL_CFG:-$CONFIG}
MASK_TYPE=${MASK_TYPE:-span}
MASK_SCHEDULE=${MASK_SCHEDULE:-uniform}
MASK_SCHEDULE_ARGS=${MASK_SCHEDULE_ARGS:-'{"b_l":0.05,"b_h":0.95}'}
SPAN_LEN=${SPAN_LEN:-64}
WHOLE_STAGE=${WHOLE_STAGE:-1}
CURRICULUM=${CURRICULUM:-1}
CUR_STAGES=${CUR_STAGES:-'[[0,2000,16,0.30],[2000,5000,32,0.60],[5000,10000,64,0.95]]'}
ROLE_MASK=${ROLE_MASK:-1}
ROLE_CONFIG=${ROLE_CONFIG:-'{"connectives":["because","despite","however","therefore","although","thus","since","while","whereas","yet","but","so","if"],"relation_verbs":["increases","reduces","decreases","inhibits","promotes","correlates","depends","drives","affects","influences","regulates","enhances","suppresses","causes","leads","results","associated","linked","modulates","triggers"],"species_keywords":["panthera","quercus","apis","daphnia","mytilus","giraffa","cervus","populus","saguaro","tigris","ilex","suber"],"number_weight":2.0,"connective_weight":3.0,"relation_weight":3.0,"entity_weight":2.0,"uppercase_bonus":0.5}'}
CANDIDATE_FOCUS=${CANDIDATE_FOCUS:-0.30}
LOSS_NUM_W=${LOSS_NUM_W:-1.0}
LOSS_NEG_W=${LOSS_NEG_W:-1.0}
CORRECTIVE_P=${CORRECTIVE_P:-0.0}
CORRECTIVE_BOOST=${CORRECTIVE_BOOST:-8.0}
MRAS_GAMMA=${MRAS_GAMMA:-0.0}
MRAS_EMA=${MRAS_EMA:-0.999}
MRAS_FLOOR=${MRAS_FLOOR:-0.3}
LR=${LR:-2.0e-4}
LR_DECAY=${LR_DECAY:-cosine}
LR_MIN_RATIO=${LR_MIN_RATIO:-0.1}
WARMUP=${WARMUP:-200}
GRAD_CLIP=${GRAD_CLIP:-1.0}
EMA_DECAY=${EMA_DECAY:-0.0}
WEIGHT_TYING=${WEIGHT_TYING:-0}
USE_ROPE=${USE_ROPE:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
VOCAB=${VOCAB:-126080}
HIDDEN=${HIDDEN:-512}
LAYERS=${LAYERS:-8}
HEADS=${HEADS:-8}
FF_MULT=${FF_MULT:-4}
N_EXPERTS=${N_EXPERTS:-1}
EXPERT_K=${EXPERT_K:-1}
SEQ_LEN=${SEQ_LEN:-768}
OUT=${OUT:-$RUNS/$TAG}
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT=${AUTO_RESUBMIT:-1}
mkdir -p "$OUT"
SIGNAL_RECEIVED=0
FINALIZED=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_P2P_DISABLE=1

# --- runtime stack: overlay+venv solo en Blackwell (sm_120 requiere torch>=2.6)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "?")
case "$GPU_NAME" in
  *"PRO 6000"*|*Blackwell*) OVL_ARGS="--overlay $BASE/blackwell-torch.overlay:ro"; PYBIN="$VENV/bin/python";;
  *) OVL_ARGS=""; PYBIN=python3;;
esac
echo "GPU detectada: $GPU_NAME -> OVL='${OVL_ARGS:-none}' PY=$PYBIN"

flag() { [ "$1" = "1" ] && echo "$2" || echo ""; }

# VRAM guard: batch8 x seq768 no cabe en ~16GB (g0-marked OOM en V100-SXM2-16G).
# Re-encola a otra ola/nodo en vez de morir en backward.
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)
MIN_VRAM_MIB=${MIN_VRAM_MIB:-20000}

finalize() {
    if [ "$FINALIZED" = "1" ]; then echo "=== finalize ya ejecutado, skip ==="; return; fi
    FINALIZED=1
    echo "=== finalize $(date -u) ==="
    if [ -f "$FLAG" ]; then
        echo "training complete; flag=$FLAG"; cat "$FLAG"; exit 0
    fi
    if [ "$AUTO_RESUBMIT" = "1" ]; then
        echo "resubmiting ola ($TAG) target=$TARGET_STEPS..."
        sbatch --export=ALL "$0"
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

if [ "$GPU_MIB" -lt "$MIN_VRAM_MIB" ]; then
    echo "VRAM ${GPU_MIB}MiB < ${MIN_VRAM_MIB}MiB -> re-encolo en otro nodo"
    finalize
fi

# dependencia suave del cache (max 4h) por si se lanzo sin --dependency
for i in $(seq 1 480); do
    [ -f "$DATA_CACHE" ] && break
    echo "esperando $DATA_CACHE ... ($i)"; sleep 30
done
[ -f "$DATA_CACHE" ] || { echo "FATAL: no existe $DATA_CACHE"; exit 1; }

# shellcheck disable=SC2086
apptainer exec --nv $OVL_ARGS "$SIF" "$PYBIN" -c "
import torch
print('GPUs:', torch.cuda.device_count())
assert torch.cuda.device_count() >= 1, 'sin GPU'
" || { echo 'GPU detect FAIL'; exit 1; }

echo "=== Lanzando G0 $TAG (steps=$TARGET_STEPS, corpus=$DATA_CACHE) $(date -u) ==="

# shellcheck disable=SC2086
apptainer exec --nv $OVL_ARGS "$SIF" \
    "$PYBIN" "$BASE/scripts/train_mdlm_moe_v2.py" \
    --data_cache "$DATA_CACHE" \
    --vocab "$VOCAB" --hidden "$HIDDEN" --layers "$LAYERS" --heads "$HEADS" --ff_mult "$FF_MULT" \
    --n_experts "$N_EXPERTS" --expert_k "$EXPERT_K" --seq_len "$SEQ_LEN" \
    --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" \
    --mask_type "$MASK_TYPE" --mask_schedule "$MASK_SCHEDULE" \
    --mask_schedule_args "$MASK_SCHEDULE_ARGS" \
    --span_len "$SPAN_LEN" \
    $(flag "$WHOLE_STAGE" --whole_stage) \
    $(flag "$CURRICULUM" --curriculum) --cur_stages "$CUR_STAGES" \
    $(flag "$ROLE_MASK" --role_mask) --role_config "$ROLE_CONFIG" \
    --candidate_focus "$CANDIDATE_FOCUS" \
    --loss_num_w "$LOSS_NUM_W" --loss_neg_w "$LOSS_NEG_W" \
    --corrective_p "$CORRECTIVE_P" --corrective_boost "$CORRECTIVE_BOOST" \
    --mras_gamma "$MRAS_GAMMA" --mras_ema "$MRAS_EMA" --mras_floor "$MRAS_FLOOR" \
    --lr "$LR" --lr_decay "$LR_DECAY" --lr_min_ratio "$LR_MIN_RATIO" \
    --warmup "$WARMUP" --max_steps "$TARGET_STEPS" \
    --grad_clip "$GRAD_CLIP" --ema_decay "$EMA_DECAY" \
    $(flag "$WEIGHT_TYING" --weight_tying) \
    $(flag "$USE_ROPE" --use_rope) \
    $([ -n "${SEED:-}" ] && echo --seed "$SEED") \
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

CKPT_DIR=$(ls -d "$OUT"/checkpoint-g* 2>/dev/null | sort -V | tail -1 || true)
if [ -z "$CKPT_DIR" ]; then
    echo "[eval] sin checkpoints en $OUT"; exit 1
fi
CKPT="$CKPT_DIR/model.pt"
MODEL_OV="{\"vocab\":$VOCAB,\"hidden\":$HIDDEN,\"layers\":$LAYERS,\"heads\":$HEADS,\"ff_mult\":$FF_MULT,\"seq_len\":$SEQ_LEN,\"n_experts\":$N_EXPERTS,\"k\":$EXPERT_K}"
echo "=== eval $CKPT con logicdiff dense/base sobre $PAIRS_DIR (ov=$MODEL_OV) ==="

# shellcheck disable=SC2086
apptainer exec --nv $OVL_ARGS "$SIF" "$PYBIN" \
    "$BASE/harness/suite_smoke_logicdiff.py" \
    --ckpt "$CKPT" --config "$EVAL_CFG" --pairs-dir "$PAIRS_DIR" \
    --out-dir "$OUT/battery_logicdiff" --device cuda \
    --model-override "$MODEL_OV" \
    --modes "base,dense" || echo "[eval] logicdiff FAILED"

# shellcheck disable=SC2086
apptainer exec --nv $OVL_ARGS "$SIF" "$PYBIN" \
    "$BASE/scripts/verdict.py" --battery "$OUT/battery_logicdiff/base" \
    --out "$OUT/battery_verdict_base.json" || echo "[eval] verdict base FAILED"
# shellcheck disable=SC2086
apptainer exec --nv $OVL_ARGS "$SIF" "$PYBIN" \
    "$BASE/scripts/verdict.py" --battery "$OUT/battery_logicdiff/dense" \
    --out "$OUT/battery_verdict_dense.json" || echo "[eval] verdict dense FAILED"

# PPL proxy: CE de denoising a fracs fijas sobre docs correctos del holdout
# (solo ctx+ok; no toca etiquetas/pares -> no contamina la metrica pairwise)
HOLDOUT_PAIRS=${HOLDOUT_PAIRS:-$RUNS/pairs_hard_v4_holdout}
apptainer exec --nv $OVL_ARGS "$SIF" "$PYBIN" \
    "$BASE/scripts/eval_ppl_proxy.py" \
    --ckpt "$CKPT" --model-override "$MODEL_OV" \
    --pairs-dir "$HOLDOUT_PAIRS" --device cuda \
    --out "$OUT/ppl_proxy.json" || echo "[eval] ppl_proxy FAILED"
echo "=== G0 $TAG DONE $(date -u) ==="
