#!/bin/bash
#SBATCH --job-name=mdlm-cuda-smoke2
#SBATCH --partition=sixhour
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=6
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=50G
#SBATCH --time=00:25:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/mdlm_cuda_smoke2_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/mdlm_cuda_smoke2_%j.err
set -euo pipefail
apptainer exec --nv --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1     /beegfs/a474r867/ecoreasoner/pytorch-cuda.sif     python3 /beegfs/a474r867/ecoreasoner/scripts/mdlm_cuda_smoke2.py
