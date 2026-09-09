#!/usr/bin/env bash
#SBATCH --job-name=moe-E2-full
#SBATCH --partition=sixhour
#SBATCH --time=05:50:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=60G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_e2full_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_e2full_%j.err

# EXPERIMENTO COMPLETO E2+E4 — 3000 steps, checkpoint cada 500, para comparar con baseline
cd /beegfs/a474r867/ecoreasoner
apptainer exec --nv pytorch-cuda.sif python3 scripts/train_mdlm_moe_expE2.py \
  --data_cache data/train_ids_v3.npy \
  --max_steps 3000 \
  --log_every 100 \
  --out_dir outputs/moe_expE2_full \
  --shared 1
echo "E2 full done rc=$?"