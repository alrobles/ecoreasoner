#!/usr/bin/env bash
#SBATCH --job-name=moe-E2v3
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=60G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_e2v3_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_e2v3_%j.err

# OLA E2+E4 v3 — Plan B: regularización por ENTROPÍA del gate + aux switch + cosine.
# Objetivo: recuperar el balance del router en capas altas (v2 terminó effN 2.49, capa 11 ~1).
# Cambios vs v2 (27452507):
#   - --ent_beta 0.1 (entropy regularization, Plan B) añadido
#   - se mantiene --aux_coeff 0.01 (switch) + cosine decay + warmup 500
#   - 6000 steps, ckpt cada 1000
#   - log de métricas: loss, aux, ent, lr, entH, effN cada 100
cd /beegfs/a474r867/ecoreasoner
apptainer exec --nv pytorch-cuda.sif python3 scripts/train_mdlm_moe_expE2.py \
  --data_cache /beegfs/a474r867/ecoreasoner/data/train_ids_v3.npy \
  --max_steps 6000 \
  --log_every 100 \
  --ckpt_every 1000 \
  --aux_coeff 0.01 \
  --ent_beta 0.1 \
  --lr_decay cosine \
  --warmup 500 \
  --lr_min_ratio 0.1 \
  --out_dir outputs/moe_expE2_v3
echo "E2 v3 done rc=$?"