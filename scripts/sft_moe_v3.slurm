#!/usr/bin/env bash
#SBATCH --job-name=sft-moe-v3
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=100G
#SBATCH --time=05:50:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/sft_moe_v3_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/sft_moe_v3_%j.err
# SFT V3 del LLaDA-MoE: generacion de CODIGO Python ejecutable.
# Dataset: 86 gold_programs SAFE de ScienceAgentBench + 14,178 ecocode L1.
# (16 golds GIS del evalset EXCLUIDOS -> eval holdout limpio).
# Corre en 2x pro6000 (venv Blackwell torch 2.7.1 — el modeling del MoE NO
# funciona en torch 2.4.1 del SIF base: CUDA device-side assert en expert_mask).
# Olas: SIGUSR1@300 + AUTO_RESUBMIT (patron bw3); el trainer guarda ckpt
# lora-g{step} en la senal y resume semi-caliente desde el ultimo (2026-08-31).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF="$BASE/pytorch-cuda.sif"
OVL="$BASE/blackwell-torch.overlay"
VENV=/bb/bwvenv
WAVE_ID="${WAVE_ID:-sft-moe-v3}"
OUT="${WAVE_OUT:-$BASE/outputs/sft_moe_v3}"
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
mkdir -p "$OUT"
SIGNAL_RECEIVED=0

# CRITICO: venv-extra primero -> transformers 4.56.2 + torchvision 0.22.1+cu128
# (sin esto, el python del bwvenv hereda el transformers/torchvision 0.19.1 del
# SIF base y muere con "operator torchvision::nms does not exist").
export PYTHONPATH="$BASE/venv-extra:${PYTHONPATH:-}"

# ---- Slurm trap (ola: SIGUSR1 300s antes del fin de walltime) --------------
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
        echo "resubmiting ola ($WAVE_ID)..."
        sbatch --export="AUTO_RESUBMIT=1,WAVE_ID=$WAVE_ID,WAVE_OUT=$OUT" \
            "$BASE/scripts/sft_moe_v3.slurm" 2>/dev/null || \
        sbatch "$BASE/scripts/sft_moe_v3.slurm"
        echo "ola siguiente enviada."
    else
        echo "auto-resubmit disabled"
    fi
    exit 42
}
handle_signal() {
    echo "=== SIGUSR1 recibido $(date -u) ==="
    pkill -USR1 -f "sft_lladamoe.py" 2>/dev/null || true
    if [ -n "${TRAIN_PID:-}" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -USR1 "$TRAIN_PID" 2>/dev/null || true
    fi
    set +e; wait "$TRAIN_PID" 2>/dev/null; train_x=$?; set -e
    echo "trainer cerrado tras senal (exit $train_x)"
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

# ---- Lanzar SFT (2x pro6000, venv Blackwell) -------------------------------
echo "=== Lanzando SFT V3 ($WAVE_ID -> $OUT) $(date -u) ==="
apptainer exec --nv --overlay "$OVL":ro "$SIF" "$VENV/bin/python" \
    "$BASE/scripts/sft_lladamoe.py" \
    --batch 8 --max-len 1536 --rank 64 --epochs 2 \
    --lr 1e-5 --warmup 300 \
    --pairs "$BASE/data/l1/sft_v3_pairs.jsonl" \
    --out "$OUT" &
TRAIN_PID=$!
set +e; wait "$TRAIN_PID"; TRAIN_X=$?; set -e
echo "=== sft exit $TRAIN_X ==="
case "$TRAIN_X" in
  0) echo "COMPLETE" > "$FLAG"; exit 0;;
  42) finalize;;
  *) if [ "$SIGNAL_RECEIVED" = "1" ]; then finalize; else
        echo "fail $TRAIN_X"; exit "$TRAIN_X"; fi;;
esac