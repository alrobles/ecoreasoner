#!/usr/bin/env bash
#SBATCH --job-name=moe-v4-bw4
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=180G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_v4_bw4_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_v4_bw4_%j.err

# =============================================================================
# EcoReasoner dLLM MoE v4 — ENTRENAMIENTO 4x Blackwell (2 nodos x 2 RTX PRO
# 6000, sm_120). MEJORA sobre bw0 (la ultima version, 1 nodo x 2, 4 exp top-1,
# corpus v4, world=2):
#
#   - 4 GPUs (2 nodos) -> world=4, batch global 16 (vs 8 de bw0)
#   - MoE 8 expertos top-2 (misma activacion 2/8=25% que 1/4, MAS capacidad
#     total de expertos -> mejor especializacion por dominio)
#   - Corpus v5 (train_ids_v5.npy, 1.118B tok) vs v4 de bw0
#   - Post-localSGD sync_every=4 + grad clip 1.0 (trainer lsgd) -> robustez
#     multi-nodo, red /4
#
# Patron de lanzamiento validado 2026-08-28:
#   - 1 contenedor por nodo (ntasks-per-node=1) + torchrun --nproc_per_node=2
#     DENTRO, para NO montar el mismo overlay ext3 en paralelo (lock exit 255).
#   - Multi-nodo: torchrun --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT para que
#     los 4 workers (2 por nodo) se encuentren via rendezvous TCP.
#   - MASTER_ADDR exportado desde el HOST con scontrol (fix multinodo validado).
#   - torch 2.7.1+cu128 en venv /bb/bwvenv dentro de blackwell-torch.overlay.
#   - trainer: train_mdlm_moe_lsgd.py (sync_every/clip) con cache v5.
#   - Ondas SIGUSR1@300 + AUTO_RESUBMIT hasta TARGET_STEPS.
# =============================================================================
set -euo pipefail

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
WAVE_ID="${WAVE_ID:-bw1}"
OUT="${WAVE_OUT:-$BASE/outputs/$WAVE_ID}"
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
TARGET_STEPS="${TARGET_STEPS:-6000}"
SYNC_EVERY="${SYNC_EVERY:-4}"
CLIP="${CLIP:-1.0}"
export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29518

mkdir -p "$OUT"

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
    # MULTI-NODO: SOLO el rank 0 (master) hace el resubmit; los otros nodos
    # salen 42 sin sbatch (evita lanzar 2 jobs duplicados).
    if [ -n "${SLURM_PROCID:-}" ] && [ "$SLURM_PROCID" != "0" ]; then
        echo "rank $SLURM_PROCID: no-resubmit (solo master resubmitea)"
        exit 42
    fi
    if [ "$AUTO_RESUBMIT" = "1" ]; then
        echo "resubmiting ola ($WAVE_ID) target=$TARGET_STEPS..."
        sbatch --export="AUTO_RESUBMIT=1,TARGET_STEPS=$TARGET_STEPS,WAVE_ID=$WAVE_ID,WAVE_OUT=$OUT,SYNC_EVERY=$SYNC_EVERY,CLIP=$CLIP" \
            "$BASE/scripts/moe_v4_bw4.slurm" 2>/dev/null || \
        sbatch "$BASE/scripts/moe_v4_bw4.slurm"
        echo "ola siguiente enviada."
    else
        echo "auto-resubmit disabled"
    fi
    exit 42
}
handle_signal() {
    echo "=== SIGUSR1 recibido $(date -u) ==="
    SIGNAL_RECEIVED=1
    # 1) Señal directa a los workers python del trainer lsgd
    pkill -USR1 -f "train_mdlm_moe_lsgd.py" 2>/dev/null || true
    # 2) También al agente torchrun (en este nodo) por si propaga él
    pkill -USR1 -f "torch.distributed.run" 2>/dev/null || true
    set +e; wait; set -e
    echo "trainer cerrado tras señal"
    finalize
}
trap handle_signal USR1
trap 'echo SIGTERM; exit 1' TERM

# verificar 2 GPUs por nodo
apptainer exec --nv --overlay "$OVL":ro "$SIF" env PYTHONDONTWRITEBYTECODE=1 "$VENV/bin/python" -c "
import torch
print('GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(' ', torch.cuda.get_device_name(i))
assert torch.cuda.device_count() >= 2
" || { echo 'GPU detect FAIL'; exit 1; }

echo "=== Lanzando MoE v4 4xBlackwell ($WAVE_ID -> $OUT, steps $TARGET_STEPS) master=$MASTER_ADDR $(date -u) ==="
# MULTI-NODO: overlay montado SOLO-lectura (:ro) -> lock COMPARTIDO entre nodos.
# Con escritura (rw), ambos nodos montan el MISMO archivo -> apptainer toma lock
# EXCLUSIVO -> el 2o nodo muere (exit 255, "in use by another process"). El venv
# /bb/bwvenv solo se lee (torch ya compilado); checkpoints/logs van a $OUT en /beegfs.
srun --export=ALL --ntasks=2 --ntasks-per-node=1 \
  apptainer exec --nv --overlay "$OVL":ro "$SIF" \
    env PYTHONDONTWRITEBYTECODE=1 "$VENV/bin/python" -m torch.distributed.run \
    --nnodes=2 --nproc_per_node=2 --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="bw4-$WAVE_ID" \
    "$BASE/scripts/train_mdlm_moe_lsgd.py" \
    --data_cache "$BASE/data/train_ids_v5_clean.npy" \
    --data "$BASE/data/train_corpus_v5.jsonl" \
    --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
    --n_experts 8 --expert_k 2 --seq_len 768 \
    --batch_size 4 --grad_accum 4 --mask_p 0.15 \
    --sync_every "$SYNC_EVERY" --clip_grad "$CLIP" \
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