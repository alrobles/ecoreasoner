#!/usr/bin/env bash
#SBATCH --job-name=nccl-test
#SBATCH --partition=sixhour
#SBATCH --time=00:08:00
#SBATCH --nodes=2 --ntasks=2 --ntasks-per-node=1
#SBATCH --cpus-per-task=4 --gres=gpu:q6000:1 --mem=16G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/nccl_test_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/nccl_test_%j.err
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
# FIX: MASTER_ADDR se calcula en el HOST (scontrol existe fuera del SIF) y se exporta.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29513
echo "=== master_addr=$MASTER_ADDR nodos=$SLURM_JOB_NODELIST ==="
srun --export=ALL -N 2 -n 2 --ntasks-per-node=1 apptainer exec --nv $BASE/pytorch-cuda.sif python3 $BASE/scripts/_nccl_test.py
echo "nccl-test done rc=$?"
