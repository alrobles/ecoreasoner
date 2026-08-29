#!/bin/bash
# =============================================================================
# moe_v4_q6000_4n.slurm — Entrenamiento MoE v4 multi-nodo en Q6000 (post-localSGD)
#
# DISEÑO (2026-08-28, preparado — NO lanzado aún):
#  - Patrón validado en moe_multi_smoke.slurm: srun + apptainer exec SIN overlay
#    (Q6000 = sm_86, el SIF base pytorch-cuda.sif ya lo soporta; el overlay
#    blackwell-torch.overlay es SOLO para Blackwell sm_120).
#  - MASTER_ADDR exportado desde el HOST con scontrol (fix multinodo validado,
#    reference moe-multinode-master-addr-fix.md) ANTES del srun.
#  - Trainer: train_mdlm_moe_lsgd.py (variante con --sync_every post-localSGD
#    + --clip_grad). NO toca train_mdlm_moe.py que usa bw0.
#  - Ondas SIGUSR1 + AUTO_RESUBMIT (patrón q35_wave/bw0): cada ola guarda
#    checkpoint y se re-lanza sola hasta TARGET_STEPS.
#  - El banco con no_sync() de DDP hace que la comunicacion sea /sync_every:
#    con 12-16 ranks el all-reduce de 3.4GB (modelo 850M) es ~17s/step a la
#    red Q6000; sync_every=8 -> ~2s/step efectivo de red.
#
# INVENTARIO Q6000 (verificado):
#   r15r05n01/r15r10n01 : 4 GPUs (32 cores)  <- los UNICOS con 4
#   r22r*  (7 nodos)     : 3 GPUs (24 cores)
#   No existen 4 nodos con 4 tarjetas -> el max homogeneo 4-nodo es 4x3=12
#   ranks (4 nodos r22) o 2x4=8 (r15). La mezcla 4+4+3+3=14 es posible
#   con --exclude de nodos concretos.
#
# USO (editar nodos/ntasks antes de lanzar, o con --export):
#   sbatch --export=ALL,NODES=r22r05n01,r22r10n01,r22r15n01,r22r20n01,\
#     TASKS=12,TASKS_PER_NODE=3,SYNC_EVERY=8,CLIP=1.0,WAVE_ID=q12,\
#     TARGET_STEPS=6000 moe_v4_q6000_4n.slurm
# =============================================================================
#SBATCH --job-name=moe-v4-q6000
#SBATCH --partition=sixhour
#SBATCH --nodes=4
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:q6000:3
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=05:50:00
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_v4_q6000_4n_%j.err
set -euo pipefail

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif          # NO overlay: Q6000 es sm_86, SIF base OK
WAVE_ID="${WAVE_ID:-q12}"
OUT="${WAVE_OUT:-$BASE/outputs/$WAVE_ID}"
FLAG=$OUT/training_complete.flag
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
TARGET_STEPS="${TARGET_STEPS:-6000}"
SYNC_EVERY="${SYNC_EVERY:-8}"       # post-localSGD: all-reduce cada 8 pasos
CLIP="${CLIP:-1.0}"                 # grad norm clip (robustez)
GRAD_ACCUM="${GRAD_ACCUM:-4}"
BATCH="${BATCH:-4}"
HIDDEN="${HIDDEN:-1024}"
LAYERS="${LAYERS:-16}"
N_EXPERTS="${N_EXPERTS:-4}"
EXPERT_K="${EXPERT_K:-1}"
SEQ_LEN="${SEQ_LEN:-768}"
DATA_CACHE="$BASE/data/train_ids_v5.npy"

mkdir -p "$OUT"
echo "=== ola $WAVE_ID en nodos: $SLURM_JOB_NODELIST | tasks=$SLURM_NTASKS sync_every=$SYNC_EVERY $(date -u)"

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
            "$BASE/scripts/moe_v4_q6000_4n.slurm" 2>/dev/null || \
        sbatch "$BASE/scripts/moe_v4_q6000_4n.slurm"
        echo "ola siguiente enviada."
    else
        echo "auto-resubmit disabled"
    fi
    exit 42
}
handle_signal() {
    echo "=== SIGUSR1 recibido $(date -u) ==="
    SIGNAL_RECEIVED=1
    # 1) Señal directa a los workers python (guarantiza checkpoint+exit 42)
    pkill -f "train_mdlm_moe_lsgd" -USR1 2>/dev/null || true
    # 2) También al agente torchrun por si propaga él
    pkill -f "torch.distributed.run" -USR1 2>/dev/null || true
    # 3) Espera corta a que los workers hagan checkpoint y salgan limpio
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
export MASTER_PORT=29512

# arena NCCL: una GPU por task (CUDA_VISIBLE_DEVICES lo fija srun+gres)
export NCCL_DEBUG=INFO

srun --export=ALL --ntasks=$SLURM_NTASKS --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
  apptainer exec --nv "$SIF" python3 scripts/train_mdlm_moe_lsgd.py \
  --data_cache "$DATA_CACHE" \
  --hidden "$HIDDEN" --layers "$LAYERS" --n_experts "$N_EXPERTS" --expert_k "$EXPERT_K" \
  --seq_len "$SEQ_LEN" --batch_size "$BATCH" --grad_accum "$GRAD_ACCUM" \
  --sync_every "$SYNC_EVERY" --clip_grad "$CLIP" \
  --max_steps "$TARGET_STEPS" --output "$OUT"

rc=$?
echo "=== srun rc=$rc === (42=ola normal, otras: exec_fail)"
if [ "$rc" = "42" ]; then finalize; fi
exit "$rc"