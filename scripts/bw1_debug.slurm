#!/usr/bin/env bash
#SBATCH --job-name=bw1-dbg
#SBATCH --partition=sixhour
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/bw1_dbg_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/bw1_dbg_%j.err

# Debug del crash "illegal memory access" de bw1 (8 exp top-2, corpus v5).
# Aisla: 1 nodo world=2, config EXACTA bw1, pocos pasos, CUDA_LAUNCH_BLOCKING=1.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF="$BASE/pytorch-cuda.sif"
OVL="$BASE/blackwell-torch.overlay"
VENV=/bb/bwvenv
cd "$BASE"
apptainer exec --nv --overlay "$OVL":ro "$SIF" env PYTHONDONTWRITEBYTECODE=1 \
  CUDA_LAUNCH_BLOCKING=1 \
  "$VENV/bin/python" -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d --rdzv_id=bw1-dbg \
  "$BASE/scripts/train_mdlm_moe_lsgd.py" \
  --data_cache "$BASE/data/train_ids_v5.npy" \
  --data "$BASE/data/train_corpus_v5.jsonl" \
  --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
  --n_experts 8 --expert_k 2 --seq_len 768 \
  --batch_size 4 --grad_accum 4 --mask_p 0.15 \
  --sync_every 4 --clip_grad 1.0 \
  --lr 2e-4 --warmup 5 --max_steps 20 \
  --output "$BASE/outputs/bw1_dbg"
echo "bw1-dbg done rc=$?"
