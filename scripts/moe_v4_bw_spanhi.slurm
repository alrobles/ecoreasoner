#!/usr/bin/env bash
#SBATCH --job-name=moe-v4-bw-spanhi
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=80G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_v4_bw_spanhi_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_v4_bw_spanhi_%j.err
# =============================================================================
# EcoReasoner dLLM MoE v4 — EXPERIMENTO SPAN RATIO ALTO (bw5_spanhi, 2026-09-04).
# Decisison usuario (04-09): ultimo ajuste del objetivo con respaldo literatura
# (T5/unified-io): mismos spans contiguos pero RATIO ALTO (~60% enmascarado vs
# 7.4% de bw4_span). Hipotesis: con la mayoria de la secuencia enmascarada, el
# modelo NO puede atajar copiando al vecino y debe reconstruir estructura desde
# contexto escaso -> sintaxis real.
#
# A/B contra bw4_span (resultado 03-09: rep4 0.003 pero word 0.19-0.34, sopa de
# subwords): MISMO warm-start limpio (bw1v5-g6000, sin pasar por bw3/bw4_span),
# MISMO corpus train_ids_v7_clean.npy, MISMA arquitectura/hiperparams, MISMO
# span_len 64. UNICA diferencia: --mask_p 1.2 (n_masked = int((768//2)*1.2) =
# 460 tok = 60% real de la secuencia; el codigo permite mask_p>1).
#
# WARM-START: copiar ANTES bw1v5/checkpoint-g6000/{model.pt,optimizer.pt} +
# state.json (step=6000) a outputs/bw5_spanhi/.
# TARGET_STEPS (absoluto) = 6000 + 1 epoch v7 (31,865) = 37,865 -> 38000.
# Criterio exito pre-registrado (mismo que bw4_span): smoke_cont sobre
# bw5_spanhi muestra rep4->~0.15 Y word_ratio->~0.435 con contexto.
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
WAVE_ID="${WAVE_ID:-bw5_spanhi}"
OUT="${WAVE_OUT:-$BASE/outputs/$WAVE_ID}"
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
TARGET_STEPS="${TARGET_STEPS:-38000}"
NPROC="${NPROC:-2}"
mkdir -p "$OUT"
SIGNAL_RECEIVED=0
FINALIZED=0
# NCCL Blackwell (2026-09-01): fix 2.26.5 + workarounds (bw3 validado).
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
        sbatch --export="AUTO_RESUBMIT=1,TARGET_STEPS=$TARGET_STEPS,WAVE_ID=$WAVE_ID,WAVE_OUT=$OUT,NPROC=$NPROC" \
            "$BASE/scripts/moe_v4_bw_spanhi.slurm" 2>/dev/null || \
        sbatch "$BASE/scripts/moe_v4_bw_spanhi.slurm"
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
for i in range(torch.cuda.device_count()):
    print(' ', torch.cuda.get_device_name(i))
assert torch.cuda.device_count() >= $NPROC
" || { echo 'GPU detect FAIL'; exit 1; }
echo "=== Lanzando MoE v4 SPAN-HI ($WAVE_ID -> $OUT, steps $TARGET_STEPS, nproc $NPROC) $(date -u) ==="
apptainer exec --nv --overlay "$OVL":ro "$SIF" \
    "$VENV/bin/python" -m torch.distributed.run \
    --nproc_per_node=$NPROC --master_port=29514 \
    "$BASE/scripts/train_mdlm_moe.py" \
    --data_cache "$BASE/data/train_ids_v7_clean.npy" \
    --data "$BASE/data/curate/train_corpus_v7_curated.jsonl" \
    --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
    --n_experts 4 --expert_k 1 --seq_len 768 \
    --batch_size 4 --grad_accum 4 --mask_p 1.2 \
    --mask_type span --span_len 64 \
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