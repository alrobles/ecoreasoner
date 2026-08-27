#!/bin/bash
#SBATCH --job-name=anal-expE2
#SBATCH --partition=sixhour
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/analyze_expE2_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/analyze_expE2_%j.err

# Revisión de activación de expertos — checkpoint e2_final del experimento E2+E4.
# CPU-only (no --gres) porque solo cargamos el modelo y hacemos forward corto.
cd /beegfs/a474r867/ecoreasoner
python3 scripts/analyze_activation_expE2.py \
  --ckpt outputs/moe_expE2_full/e2_final.pt \
  --data /beegfs/a474r867/ecoreasoner/data/train_ids_v3.npy \
  --n_samples 4
echo "analyze done rc=$?"