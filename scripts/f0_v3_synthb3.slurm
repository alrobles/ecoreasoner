#!/usr/bin/env bash
#SBATCH --job-name=f0-span-v3-synthb3
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=80G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/f0_span_v3_synthb3_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/f0_span_v3_synthb3_%j.err
# B3: receta B1 EXACTA (f0-span-v3-candfocus) pero con el corpus mezclado
# skeleton_v2 + 30K docs sinteticos FLD (deduccion procedural con near-miss).
# Aisla el efecto del INPUT (contenido inferencial) vs B1: si dense L3 sube y
# los subtipos number/negation se reparan (>0.6), el input era el cuello.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
RUNS=$BASE/runs

TAG=f0-span-v3-synthb3
DATA_CACHE=/beegfs/a474r867/ecoreasoner/data/train_ids_b3.npz
CONFIG=/beegfs/a474r867/ecoreasoner/harness/configs/f0-span-v3-role.yaml
TARGET_STEPS=10000
PAIRS_DIR=/beegfs/a474r867/ecoreasoner/runs/pairs_hard_v3_eval
EVAL_CFG=/beegfs/a474r867/ecoreasoner/harness/configs/f0-span-v3-role.yaml
MASK_TYPE=span
MASK_SCHEDULE=uniform
MASK_SCHEDULE_ARGS='{"b_l":0.05,"b_h":0.95}'
SPAN_LEN=64
WHOLE_STAGE=1
CURRICULUM=1
CUR_STAGES='[[0,2000,16,0.30],[2000,5000,32,0.60],[5000,10000,64,0.95]]'
ROLE_MASK=1
ROLE_CONFIG='{"connectives":["because","despite","however","therefore","although","thus","since","while","whereas","yet","but","so","if"],"relation_verbs":["increases","reduces","decreases","inhibits","promotes","correlates","depends","drives","affects","influences","regulates","enhances","suppresses","causes","leads","results","associated","linked","modulates","triggers"],"species_keywords":["panthera","quercus","apis","daphnia","mytilus","giraffa","cervus","populus","saguaro","tigris","ilex","suber"],"number_weight":2.0,"connective_weight":3.0,"relation_weight":3.0,"entity_weight":2.0,"uppercase_bonus":0.5}'
CANDIDATE_FOCUS=0.30
LR=2.0e-4
LR_DECAY=cosine
LR_MIN_RATIO=0.1
WARMUP=200
GRAD_CLIP=1.0
EMA_DECAY=0.0
WEIGHT_TYING=0
USE_ROPE=0
BATCH_SIZE=8
GRAD_ACCUM=2
VOCAB=126080
HIDDEN=512
LAYERS=8
HEADS=8
FF_MULT=4
N_EXPERTS=1
EXPERT_K=1
SEQ_LEN=768
OUT=$RUNS/$TAG
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT=1
mkdir -p "$OUT"
SIGNAL_RECEIVED=0
FINALIZED=0
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_P2P_DISABLE=1

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

apptainer exec --nv --overlay "$OVL":ro "$SIF" "$VENV/bin/python" -c "
import torch
print('GPUs:', torch.cuda.device_count())
assert torch.cuda.device_count() >= 1, 'sin GPU'
" || { echo 'GPU detect FAIL'; exit 1; }

echo "=== Lanzando B3 synthb3 $TAG (steps=$TARGET_STEPS, corpus=$DATA_CACHE) $(date -u) ==="

apptainer exec --nv --overlay "$OVL":ro "$SIF" \
    "$VENV/bin/python" "$BASE/scripts/train_mdlm_moe_v2.py" \
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

CKPT_DIR=$(ls -d "$OUT"/checkpoint-g* 2>/dev/null | sort -V | tail -1 || true)
if [ -z "$CKPT_DIR" ]; then
    echo "[eval] sin checkpoints en $OUT"; exit 1
fi
CKPT="$CKPT_DIR/model.pt"
echo "=== eval $CKPT con logicdiff dense/base sobre $PAIRS_DIR ==="

apptainer exec --nv --overlay "$OVL":ro "$SIF" "$VENV/bin/python" \
    "$BASE/harness/suite_smoke_logicdiff.py" \
    --ckpt "$CKPT" --config "$EVAL_CFG" --pairs-dir "$PAIRS_DIR" \
    --out-dir "$OUT/battery_logicdiff" --device cuda \
    --modes "base,dense" || echo "[eval] logicdiff FAILED"

apptainer exec --nv --overlay "$OVL":ro "$SIF" "$VENV/bin/python" \
    "$BASE/scripts/verdict.py" --battery "$OUT/battery_logicdiff/base" \
    --out "$OUT/battery_verdict_base.json" || echo "[eval] verdict base FAILED"
apptainer exec --nv --overlay "$OVL":ro "$SIF" "$VENV/bin/python" \
    "$BASE/scripts/verdict.py" --battery "$OUT/battery_logicdiff/dense" \
    --out "$OUT/battery_verdict_dense.json" || echo "[eval] verdict dense FAILED"
echo "=== B3 synthb3 $TAG DONE $(date -u) ==="