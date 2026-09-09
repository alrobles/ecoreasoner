#!/usr/bin/env bash
#SBATCH --job-name=moe-E2-smoke
#SBATCH --partition=sixhour
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=60G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_e2_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_e2_%j.err

# EXPERIMENTO E2+E4 — smoke-train en 1 GPU Q6000
cd /beegfs/a474r867/ecoreasoner
apptainer exec --nv pytorch-cuda.sif python3 scripts/train_mdlm_moe_expE2.py \
  --data_cache data/train_ids_v3.npy \
  --max_steps 300 \
  --log_every 25 \
  --out_dir outputs/moe_expE2 \
  --shared 1
echo "E2 smoke done rc=$?"