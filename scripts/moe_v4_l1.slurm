#!/usr/bin/env bash
#SBATCH --job-name=moe-v4-l1
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=80G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_v4_l1_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_v4_l1_%j.err

# =============================================================================
# EcoReasoner L1 — Adaptacion a tool-use (masked-diffusion sobre trazas).
# Continua desde el checkpoint final de bw1v5 con el dataset MIXTO
# (L1 tool-use + ancla v6), LR baja para no destruir el conocimiento L0.
# Trainer VANILLA (estable en DDP world=2 Blackwell).
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
WAVE_ID="${WAVE_ID:-bw2}"
OUT="${WAVE_OUT:-$BASE/outputs/$WAVE_ID}"
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
TARGET_STEPS="${TARGET_STEPS:-1500}"
mkdir -p "$OUT"

SIGNAL_RECEIVED=0
FINALIZED=0
finalize() {
    if [ "$FINALIZED" = "1" ]; then return; fi
    FINALIZED=1
    echo "=== finalize $(date -u) ==="
    if [ -f "$FLAG" ]; then echo "complete"; cat "$FLAG"; exit 0; fi
    if [ "$AUTO_RESUBMIT" = "1" ]; then
        sbatch --export="AUTO_RESUBMIT=1,TARGET_STEPS=$TARGET_STEPS,WAVE_ID=$WAVE_ID,WAVE_OUT=$OUT" \
            "$BASE/scripts/moe_v4_l1.slurm" 2>/dev/null || true
    fi
    exit 42
}
handle_signal() {
    SIGNAL_RECEIVED=1
    pkill -USR1 -f "train_mdlm_moe.py" 2>/dev/null || true
    set +e; wait "$TRAIN_PID" 2>/dev/null; train_x=$?; set -e
    finalize
}
trap handle_signal USR1
trap 'exit 1' TERM

apptainer exec --nv --overlay "$OVL":ro "$SIF" "$VENV/bin/python" -c "
import torch
assert torch.cuda.device_count() >= 2
" || { echo 'GPU detect FAIL'; exit 1; }

echo "=== Lanzando MoE L1 ($WAVE_ID -> $OUT, steps $TARGET_STEPS) $(date -u) ==="
apptainer exec --nv --overlay "$OVL":ro "$SIF" \
    "$VENV/bin/python" -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29512 \
    "$BASE/scripts/train_mdlm_moe.py" \
    --data_cache "$BASE/data/l1/train_ids_l1mix.npy" \
    --data "$BASE/data/l1/train_corpus_l1mix.jsonl" \
    --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
    --n_experts 4 --expert_k 1 --seq_len 768 \
    --batch_size 4 --grad_accum 4 --mask_p 0.15 \
    --lr 1e-4 --warmup 200 --max_steps "$TARGET_STEPS" \
    --output "$OUT" &
TRAIN_PID=$!
set +e; wait "$TRAIN_PID"; TRAIN_X=$?; set -e
echo "=== train exit $TRAIN_X ==="
case "$TRAIN_X" in
  0) echo "COMPLETE" > "$FLAG"; exit 0;;
  42) finalize;;
  *) if [ "$SIGNAL_RECEIVED" = "1" ]; then finalize; else exit "$TRAIN_X"; fi;;
esac