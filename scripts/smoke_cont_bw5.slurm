#!/usr/bin/env bash
#SBATCH --job-name=smoke-bw5
#SBATCH --partition=sixhour
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=80G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/smoke_bw5_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/smoke_bw5_%j.err
# =============================================================================
# Smoke de CONTINUACION para bw5_spanhi (2026-09-08). bw5 completó 38,000 steps
# (flag COMPLETE = checkpoint-g38000). Este es el tercer punto de la curva de
# ablación del objetivo (span ratio ALTO 60%). Criterio pre-registrado (el mismo
# que bw4_span): con contexto, rep4->~0.15 Y word_ratio->~0.435 (baseline texto
# real word=0.435, rep4=0.153).
#
# Copia de smoke_cont_span.slurm con RUN=bw5_spanhi y el checkpoint final.
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
RUN=bw5_spanhi
CKPT_STEP="${CKPT_STEP:-38000}"
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_P2P_DISABLE=1

CKPT="$BASE/outputs/$RUN/checkpoint-g$CKPT_STEP"
if [ ! -f "$CKPT/model.pt" ]; then
    echo "[fatal] checkpoint $CKPT/model.pt no existe"
    exit 1
fi
echo "=== smoke_cont bw5_spanhi ckpt=$CKPT_STEP ($(hostname)) $(date -u) ==="
apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python \
    "$BASE/scripts/smoke_cont_bw3.py" \
    --ckpt "$CKPT" \
    --n-docs 6 --ctx-lens 0,64,200,400 --n-gen 180 --temps 0.7,0.9 \
    --steps 64,128 \
    --out "$BASE/data/l1/smoke_cont_bw5_g${CKPT_STEP}.json"
echo "=== SMOKE BW5 exit=$? $(date -u) ==="