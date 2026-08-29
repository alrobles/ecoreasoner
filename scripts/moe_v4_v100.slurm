#!/bin/bash
# =============================================================================
# moe_v4_v100.slurm — Entrenamiento MoE v4 multi-nodo en V100 (16GB) — VALIDACION
#                      DE ESCALADO MULTINODO REAL (32 ranks) — config compacta
#
# OBJETIVO (2026-08-29): romper el techo de ranks multi-nodo usando el pool
# V100 (36 tarjetas, el unico disponible HOY: r12r08/r12r10 idle, 14 mas mix).
# Config COMPACTA fp32 (~450M params) para que quepa en 16GB SIN AMP (V100
# sm_70 NO soporta bf16; el trainer base corre fp32 puro).
#
#   Memoria 450M fp32: pesos 1.8G + grads 1.8G + AdamW 3.6G = ~7.2G + act ->
#   cabe holgado en 16GB con batch 4.
#
# Post-localSGD ya integrado en train_mdlm_moe_lsgd.py (--sync_every, --clip).
# V100 NO soporta bf16 pero el trainer NO lo usa (fp32) -> SIF base OK sin overlay.
# Patron ondas SIGUSR1 + AUTO_RESUBMIT (igual que moe_v4_q6000_4n).
#
# USO (defaults = 16 nodos x 2 GPU = 32 ranks, config compacta):
#   sbatch --export=WAVE_ID=v100-32,TARGET_STEPS=6000 moe_v4_v100.slurm
# Para 18 nodos x 2 = 36 ranks:
#   sbatch --export=WAVE_ID=v100-36,NODES=18 mintasks=36 moe_v4_v100.slurm
# =============================================================================
#SBATCH --job-name=moe-v4-v100
#SBATCH --partition=sixhour
#SBATCH --nodes=16
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=05:50:00
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_v4_v100_%j.err
set -euo pipefail

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif          # sin overlay: V100 sm_70, SIF base OK
WAVE_ID="${WAVE_ID:-v100-32}"
OUT="${WAVE_OUT:-$BASE/outputs/$WAVE_ID}"
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
TARGET_STEPS="${TARGET_STEPS:-6000}"
SYNC_EVERY="${SYNC_EVERY:-8}"       # post-localSGD: all-reduce cada 8 pasos
CLIP="${CLIP:-1.0}"                 # grad norm clip
GRAD_ACCUM="${GRAD_ACCUM:-4}"
BATCH="${BATCH:-4}"                 # por rank
# config COMPACTA (cabe en 16GB fp32)
HIDDEN="${HIDDEN:-768}"
LAYERS="${LAYERS:-12}"
HEADS="${HEADS:-12}"
N_EXPERTS="${N_EXPERTS:-4}"
EXPERT_K="${EXPERT_K:-1}"
FF_MULT="${FF_MULT:-4}"
SEQ_LEN="${SEQ_LEN:-512}"
DATA_CACHE="$BASE/data/train_ids_v5.npy"

mkdir -p "$OUT"
echo "=== ola $WAVE_ID nodos=$SLURM_JOB_NODELIST tasks=$SLURM_NTASKS sync=$SYNC_EVERY hidden=$HIDDEN lyr=$LAYERS seq=$SEQ_LEN $(date -u)"

SIGNAL_RECEIVED=0
FINALIZED=0
finalize() {
    if [ "$FINALIZED" = "1" ]; then echo "=== finalize ya ejecutado, skip ==="; return; fi
    FINALIZED=1
    echo "=== finalize $(date -u) ==="
    if [ -f "$FLAG" ]; then
        echo "training complete; flag=$FLAG"; cat "$FLAG"; exit 0
    fi
    if [ "$AUTO_RESUBMIT" = "1" ]; then
        echo "resubmiting ola ($WAVE_ID) target=$TARGET_STEPS..."
        sbatch --export="ALL,NODES=$SLURM_JOB_NODELIST,AUTO_RESUBMIT=1,TARGET_STEPS=$TARGET_STEPS,WAVE_ID=$WAVE_ID,WAVE_OUT=$OUT,SYNC_EVERY=$SYNC_EVERY,CLIP=$CLIP" \
            "$BASE/scripts/moe_v4_v100.slurm" 2>/dev/null || \
        sbatch "$BASE/scripts/moe_v4_v100.slurm"
        echo "ola siguiente enviada."
    else
        echo "auto-resubmit disabled"
    fi
    exit 42
}
handle_signal() {
    echo "=== SIGUSR1 recibido $(date -u) ==="
    SIGNAL_RECEIVED=1
    pkill -f "train_mdlm_moe_lsgd" -USR1 2>/dev/null || true
    pkill -f "torch.distributed.run" -USR1 2>/dev/null || true
    for i in $(seq 1 30); do
        if ! pgrep -f "train_mdlm_moe_lsgd" >/dev/null 2>&1; then break; fi
        sleep 2
    done
    finalize
}
trap handle_signal USR1
trap finalize EXIT TERM

echo "=== MASTER_ADDR: $(scontrol show hostname "$SLURM_JOB_NODELIST" | head -1) ==="
export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29515
export NCCL_DEBUG=INFO

# OJO: 16 nodos pueden tardar en arrancar todos los ranks; NCCL_DEBUG=INFO en
# la primera ola para ver el rendezvous de los 32 ranks.

srun --export=ALL --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
  apptainer exec --nv "$SIF" python3 "$BASE/scripts/train_mdlm_moe_lsgd.py" \
  --data "$BASE/data/sci_v2_B1_consolidated.jsonl" \
  --data_cache "$DATA_CACHE" \
  --hidden "$HIDDEN" --layers "$LAYERS" --heads "$HEADS" \
  --n_experts "$N_EXPERTS" --expert_k "$EXPERT_K" --ff_mult "$FF_MULT" \
  --seq_len "$SEQ_LEN" --batch_size "$BATCH" --grad_accum "$GRAD_ACCUM" \
  --sync_every "$SYNC_EVERY" --clip_grad "$CLIP" \
  --max_steps "$TARGET_STEPS" --output "$OUT"

rc=$?
echo "=== srun rc=$rc === (42=ola normal, otras: exec_fail)"
if [ "$rc" = "42" ]; then finalize; fi
exit "$rc"