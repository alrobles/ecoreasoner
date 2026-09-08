#!/usr/bin/env bash
#SBATCH --job-name=f1-hetero
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/f1_hetero_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/f1_hetero_%j.err
# =============================================================================
# F1-HETERO: entrenamiento multi-GPU HETEROGENEO (paper-1 systems) con el dLLM
# micro dense y trainer train_mdlm_moe_hetero.py (--autosize).
#
# Cada job = UN nodo de UNA familia (A100/Q6000/L40). Todos los jobs del pool
# se lanzan con el MAPEO de claves del TODO (juntos) -> torchrun hace rendezvous
# TCP en MASTER_ADDR:MASTER_PORT. RANK global lo setea torchrun.
#
# Env (--export de TODOS los jobs del run):
#   NODE_RANK   indice global del NODO (0,1,2,...). LOS jobs lo comparten.
#   NPROC       GPUs de ESTE nodo (--gres decide la familia).
#   MASTER_ADDR hostname del nodo rank 0 (primer job A100).
#   MASTER_PORT puerto comun (e.g. 29520).
#   NJOBS       numero TOTAL de jobs/nodos del run (--nnodes).
#   TARGET_STEPS  steps absolutos de este run (todos los jobs igual).
#   DATA_CACHE  npy de tokens (esqueleto para F1).
#   GLOBAL_BATCH tokens objetivo por optimizer step (default 65536 en tokens).
#   TAG        tag del run (runs/<TAG>).
#   GRES_FAMILY familia de GPU (a100/l40/q6000). Se pasa via --export y --gres.
#   AUTO_RESUBMIT 1 por defecto; 0 desactiva reenvio al finalizar ola.
#
# Uso (por cada nodo del pool):
#   sbatch --gres=gpu:a100:3 --export="NODE_RANK=0,NPROC=3,NJOBS=3,...,GRES_FAMILY=a100" scripts/f1_hetero.slurm
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
VENV=/bb/bwvenv
NODE_RANK="${NODE_RANK:?NODE_RANK requerido}"
NPROC="${NPROC:?NPROC requerido}"
NJOBS="${NJOBS:?NJOBS requerido}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR requerido}"
MASTER_PORT="${MASTER_PORT:-29520}"
TARGET_STEPS="${TARGET_STEPS:-200}"
DATA_CACHE="${DATA_CACHE:?DATA_CACHE requerido}"
GLOBAL_BATCH="${GLOBAL_BATCH:-65536}"
TAG="${TAG:-f1-hetero}"
AUTO_RESUBMIT="${AUTO_RESUBMIT:-1}"
GRES_FAMILY="${GRES_FAMILY:-}"
GRES_ARG="gpu:${GRES_FAMILY}:${NPROC}"
OUT=$BASE/runs/$TAG
mkdir -p "$OUT"
FLAG=$OUT/training_complete.flag
FINALIZED=0
SIGNAL_RECEIVED=0
TRAIN_PID=""
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_P2P_DISABLE=1
# 2026-09-07: el L40 (r32r25n01) es noib; NCCL elegia IB en el master A100 y
# el L40 fallaba (NCCLUtils.hpp:275 system call failed). Forzar TCP/Ethernet
# comun a todas las familias. NCCL_SOCKET_IFNAME: probar lo que exista.
for iface in eth0 eth1 ib0 ens3 eno1; do
    if ip link show "$iface" >/dev/null 2>&1; then
        export NCCL_SOCKET_IFNAME="$iface"
        break
    fi
done
export NCCL_IB_DISABLE=1
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<auto>} IB_DISABLE=$NCCL_IB_DISABLE"

finalize() {
    [ "$FINALIZED" -eq 1 ] && return
    FINALIZED=1
    if [ -f "$FLAG" ] && [ "$(cat "$FLAG" 2>/dev/null)" = "COMPLETE" ]; then
        exit 0
    fi
    if [ "$AUTO_RESUBMIT" -eq 1 ]; then
        echo "=== reenviando job (ola) $(date -u) ===" | tee -a "$OUT/signal.log" 2>/dev/null || true
        sbatch --gres="$GRES_ARG" \
            --export="NODE_RANK=$NODE_RANK,NPROC=$NPROC,NJOBS=$NJOBS,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT,TARGET_STEPS=$TARGET_STEPS,DATA_CACHE=$DATA_CACHE,GLOBAL_BATCH=$GLOBAL_BATCH,TAG=$TAG,AUTO_RESUBMIT=1,GRES_FAMILY=$GRES_FAMILY" \
            "$0" || true
    fi
    exit 42
}

handle_signal() {
    echo "=== SIGUSR1 recibido $(date -u) ===" | tee -a "$OUT/signal.log" 2>/dev/null || true
    SIGNAL_RECEIVED=1
    if [ -n "${TRAIN_PID:-}" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -USR1 "$TRAIN_PID" 2>/dev/null || pkill -USR1 -f train_mdlm_moe_hetero.py 2>/dev/null || true
    else
        pkill -USR1 -f train_mdlm_moe_hetero.py 2>/dev/null || true
    fi
    if [ -n "${TRAIN_PID:-}" ]; then
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
    finalize
}

trap 'handle_signal' USR1
trap 'echo SIGTERM; exit 1' TERM

echo "=== F1-HETERO nodo_rank=$NODE_RANK nproc=$NPROC njobs=$NJOBS master=$MASTER_ADDR:$MASTER_PORT gres=$GRES_ARG $(hostname) $(date -u) ==="
apptainer exec --nv "$SIF" python3 -c "
import torch
print('GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(' ', torch.cuda.get_device_name(i), round(p.total_memory/1e9), 'GB')
assert torch.cuda.device_count() >= $NPROC
"
echo "=== lanzando torchrun multi-job (autosize, esqueleto) ==="
apptainer exec --nv "$SIF" python3 -m torch.distributed.run \
    --nnodes="$NJOBS" --nproc_per_node="$NPROC" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    "$BASE/scripts/train_mdlm_moe_hetero.py" \
    --data_cache "$DATA_CACHE" \
    --data "$DATA_CACHE" \
    --vocab 126080 --hidden 512 --layers 8 --heads 8 --ff_mult 4 \
    --n_experts 1 --expert_k 1 --seq_len 768 \
    --autosize --global-batch "$GLOBAL_BATCH" --grad_accum 4 \
    --mask_type span --span_len 64 --mask_p 0.15 \
    --lr 2e-4 --warmup 100 --max_steps "$TARGET_STEPS" \
    --output "$OUT" \
    > "$OUT/train_rank$NODE_RANK.log" 2>&1 &
TRAIN_PID=$!
set +e
wait "$TRAIN_PID"
TRAIN_X=$?
set -e

if [ "$TRAIN_X" -eq 0 ]; then
    echo "COMPLETE" > "$FLAG"
    echo "=== F1-HETERO nodo $NODE_RANK exit=0 $(date -u) ==="
    exit 0
elif [ "$TRAIN_X" -eq 42 ]; then
    echo "=== F1-HETERO nodo $NODE_RANK exit=42 (ola) $(date -u) ==="
    finalize
elif [ "$SIGNAL_RECEIVED" -eq 1 ]; then
    echo "=== F1-HETERO nodo $NODE_RANK exit=$TRAIN_X signal $(date -u) ==="
    finalize
else
    echo "=== F1-HETERO nodo $NODE_RANK exit=$TRAIN_X $(date -u) ==="
    exit "$TRAIN_X"
fi
