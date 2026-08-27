#!/usr/bin/env bash
#SBATCH --job-name=moe-E2v2
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=60G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_e2v2_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_e2v2_%j.err

# OLA E2+E4 v2 — loss aux de balance de carga + LR cosine decay.
# Objetivo: atacar el colapso de router en capas altas detectado en el análisis
# de activación (EFF_N 7.6 global pero capa 11 ~1 experto efectivo, entH 0.42).
# Cambios vs e2 full (27441904):
#   - --aux_coeff 0.01 (Switch-Transformer load-balance aux)
#   - --lr_decay cosine + --warmup 500 + --lr_min_ratio 0.1
#   - 6000 steps (doble), ckpt cada 1000 (menos disco, 2.9GB c/u)
#   - log de métricas de router (entH, effN) cada 100
cd /beegfs/a474r867/ecoreasoner
apptainer exec --nv pytorch-cuda.sif python3 scripts/train_mdlm_moe_expE2.py \
  --data_cache /beegfs/a474r867/ecoreasoner/data/train_ids_v3.npy \
  --max_steps 6000 \
  --log_every 100 \
  --ckpt_every 1000 \
  --aux_coeff 0.01 \
  --lr_decay cosine \
  --warmup 500 \
  --lr_min_ratio 0.1 \
  --out_dir outputs/moe_expE2_v2
echo "E2 v2 done rc=$?"