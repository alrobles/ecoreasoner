#!/usr/bin/env bash
#SBATCH --job-name=moe-E2v4
#SBATCH --partition=sixhour
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:q6000:4
#SBATCH --mem=60G
#SBATCH --time=05:50:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_e2v4_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_e2v4_%j.err

# OLA E2+E4 v4 — MODELO MÁS POTENTE (hidden 1024, 16 layers, 12 expertos, ~2B total/595M act)
# Entrenado con DDP en 8× Q6000 (2 nodos × 4 GPU, como Fase A) sobre el dataset v4
# (corpus v3 + 1.74M PMC full-text, ~4-5B tokens).
# REQUISITO: train_ids_v4.npy debe existir (pre-tokenize v4 completado).
# Cambios vs v3 (27463176):
#   - escala: hidden 1024, layers 16, heads 16, n_experts 12 (memtest: 2072M total/595M act)
#   - dataset v4: --data_cache data/train_ids_v4.npy
#   - Plan B activo: --aux_coeff 0.01 + --ent_beta 0.1 + cosine
#   - DDP 8 ranks (escala multi-GPU)
#   - batch 1/GPU + grad_accum 4 (memoria), steps: 6000 (o más si datos lo piden)
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
DATA="$BASE/data/train_ids_v4.npy"
[ -f "$DATA" ] || { echo "FALTA train_ids_v4.npy — pre-tokenize v4 no completado"; exit 1; }

srun --export=ALL -n 8 apptainer exec --nv pytorch-cuda.sif \
  python3 scripts/train_mdlm_moe_expE2.py \
  --data_cache "$DATA" \
  --max_steps 6000 \
  --log_every 100 \
  --ckpt_every 1000 \
  --hidden 1024 \
  --layers 16 \
  --heads 16 \
  --ff_mult 4 \
  --n_experts 12 \
  --expert_k 1 \
  --seq_len 768 \
  --batch_size 1 \
  --grad_accum 4 \
  --aux_coeff 0.01 \
  --ent_beta 0.1 \
  --lr_decay cosine \
  --warmup 800 \
  --lr_min_ratio 0.1 \
  --out_dir outputs/moe_expE2_v4
echo "E2 v4 done rc=$?"