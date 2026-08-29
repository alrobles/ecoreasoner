#!/bin/bash
#SBATCH --job-name=bench-gpu
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:12:00
#SBATCH --gres=gpu:q6000:1
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/bench_gpu_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/bench_gpu_%j.err
# =============================================================================
# bench_gpu.slurm — mide TFLOPS reales de UNA GPU del tipo en --gres.
# El gres se fija con sbatch --gres (sobrescribe el header):
#   sbatch --gres=gpu:a100:1 -w r31r05n01 bench_gpu.slurm
#   sbatch --gres=gpu:l40:1  -w r32r25n01 bench_gpu.slurm
# El srun NO lleva --gres (el gres es del JOB, no del step; --gres en srun
# rompe con "Invalid generic resource (gres) specification").
# =============================================================================
set -euo pipefail
LABEL="${LABEL:-unknown}"
echo "=== bench GRES=$SLURM_JOB_GPUS NODE=$SLURM_JOB_NODELIST label=$LABEL $(date -u) ==="
srun --export=ALL \
  apptainer exec --nv /beegfs/a474r867/ecoreasoner/pytorch-cuda.sif \
  python3 /beegfs/a474r867/ecoreasoner/scripts/bench_tflops.py
echo "=== DONE $LABEL ==="