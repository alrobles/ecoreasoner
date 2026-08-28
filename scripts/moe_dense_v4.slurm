#!/usr/bin/env bash
#SBATCH --job-name=moe-dense-v4
#SBATCH --partition=sixhour
#SBATCH --nodes=4
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:q6000:3
#SBATCH --mem=60G
#SBATCH --time=05:50:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_dense_v4_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_dense_v4_%j.err

# CONTROL DENSE v4 — MISMO modelo/trainer expE2 que el MoE v4, pero n_experts=1.
# shared=1 (experto compartido) para equiparar ACTIVOS con el MoE v4 (~595M).
# Comparación válida: mismo coste activo, evalúa si el MoE (especialización)
# aporta sobre un dense de la misma arquitectura sobre el MISMO dataset v4.
# DDP 12 ranks (4 nodos × 3 Q6000 = 12 GPU), dataset train_ids_v4.npy.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
DATA="$BASE/data/train_ids_v4.npy"
[ -f "$DATA" ] || { echo "FALTA train_ids_v4.npy"; exit 1; }

srun --export=ALL -n 12 apptainer exec --nv pytorch-cuda.sif \
  python3 scripts/train_mdlm_moe_expE2.py \
  --data_cache "$DATA" \
  --hidden 1024 \
  --layers 16 \
  --heads 16 \
  --ff_mult 4 \
  --n_experts 1 \
  --expert_k 1 \
  --shared 1 \
  --seq_len 768 \
  --batch_size 1 \
  --grad_accum 4 \
  --aux_coeff 0 \
  --ent_beta 0 \
  --lr_decay cosine \
  --warmup 800 \
  --lr_min_ratio 0.1 \
  --max_steps 6000 \
  --out_dir outputs/moe_dense_v4_ctl
echo "DENSE v4 (expE2, shared=1) done rc=$?"