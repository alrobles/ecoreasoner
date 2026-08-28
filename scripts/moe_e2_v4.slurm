#!/usr/bin/env bash
#SBATCH --job-name=moe-E2v4
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:q6000:3
#SBATCH --mem=60G
#SBATCH --time=05:50:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_e2v4_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_e2v4_%j.err

# OLA E2+E4 v4 — SWEET SPOT (hidden 1024, 16 layers, 2 expertos, ~729M total/595M act)
# Desde sweep_mem: ne=2 da peak 15.0GB (margen 8.5GB) -> aguanta overhead DDP real.
# 3-4 expertos OOM en DDP real (margen insuficiente). Activos 595M = max cabible confiable.
# DDP 12 ranks, batch 1/GPU + grad_accum 2 + expandable_segments (anti-OOM).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
DATA="$BASE/data/train_ids_v4.npy"
[ -f "$DATA" ] || { echo "FALTA train_ids_v4.npy — pre-tokenize v4 no completado"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun --export=ALL -n 3 apptainer exec --nv pytorch-cuda.sif \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/train_mdlm_moe_expE2.py \
  --data_cache "$DATA" \
  --max_steps 6000 \
  --log_every 100 \
  --ckpt_every 1000 \
  --hidden 1024 \
  --layers 16 \
  --heads 16 \
  --ff_mult 4 \
  --n_experts 2 \
  --expert_k 1 \
  --seq_len 768 \
  --batch_size 1 \
  --grad_accum 2 \
  --aux_coeff 0.01 \
  --ent_beta 0.1 \
  --lr_decay cosine \
  --warmup 800 \
  --lr_min_ratio 0.1 \
  --out_dir outputs/moe_expE2_v4
echo "E2 v4 done rc=$?"