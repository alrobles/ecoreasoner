#!/usr/bin/env bash
#SBATCH --job-name=moe-v4-bw-span
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=80G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_v4_bw_span_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_v4_bw_span_%j.err

# =============================================================================
# EcoReasoner dLLM MoE v4 — EXPERIMENTO SPAN-MASKING (bw4_span, 2026-09-02).
# A/B contra bw3 (random-mask): MISMO warm-start, MISMO corpus v7_clean,
# MISMA arquitectura/hiperparams. UNICA diferencia: --mask_type span
# (bloques contiguos ~64 tok) vs random disperso (historico).
#
# MOTIVO (diagnostico smoke_cont_bw3_g77051 2026-09-02): con contexto real de
# 64-400 tok el modelo colapsa en repeticiones del vecino mas probable
# (rep4 0.20-0.41 vs 0.153 en texto real); la mascara dispersa 15% premia
# autocorrelacion local (copiar el token adyacente reduce el loss sin aprender
# sintaxis). Los spans contiguos fuerzan reconstruir REGIONES desde contexto
# lejano -> coherencia estructural.
#
# WARM-START: copiar ANTES el checkpoint de bw3 mas reciente con model.pt +
# optimizer.pt + state.json (step absoluto) a outputs/bw4_span/.
# TARGET_STEPS (absoluto) = step_warm + 1 epoch v7 (31,865):
#   aprox 77,400 + 31,865 = 109,265 -> 110,000 (1 epoch de prueba; evaluar
#   con smoke_cont_bw3.py y smoke_l1.py; si hay estructura -> seguir).
# =============================================================================
set -euo pipefail

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
WAVE_ID="${WAVE_ID:-bw4_span}"
OUT="${WAVE_OUT:-$BASE/outputs/$WAVE_ID}"
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
TARGET_STEPS="${TARGET_STEPS:-110000}"
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
            "$BASE/scripts/moe_v4_bw_span.slurm" 2>/dev/null || \
        sbatch "$BASE/scripts/moe_v4_bw_span.slurm"
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

echo "=== Lanzando MoE v4 SPAN ($WAVE_ID -> $OUT, steps $TARGET_STEPS, nproc $NPROC) $(date -u) ==="
apptainer exec --nv --overlay "$OVL":ro "$SIF" \
    "$VENV/bin/python" -m torch.distributed.run \
    --nproc_per_node=$NPROC --master_port=29513 \
    "$BASE/scripts/train_mdlm_moe.py" \
    --data_cache "$BASE/data/train_ids_v7_clean.npy" \
    --data "$BASE/data/curate/train_corpus_v7_curated.jsonl" \
    --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
    --n_experts 4 --expert_k 1 --seq_len 768 \
    --batch_size 4 --grad_accum 4 --mask_p 0.15 \
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