#!/usr/bin/env bash
#SBATCH --job-name=nccl-test
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/nccl_test_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/nccl_test_%j.err
# Mini-test: all-reduce real en 2 GPUs tras actualizar NCCL a 2.26.5.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
apptainer exec --nv --overlay "$BASE/blackwell-torch.overlay":ro "$BASE/pytorch-cuda.sif" \
  /bb/bwvenv/bin/python -m torch.distributed.run --nproc_per_node=2 --master_port=29517 \
  "$BASE/scripts/nccl_test.py"
echo "nccl test rc=$?"