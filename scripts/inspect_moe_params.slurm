#!/usr/bin/env bash
#SBATCH --job-name=insp-moe
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/insp_moe_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/insp_moe_%j.err
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
export PYTHONPATH="$BASE/venv-extra:${PYTHONPATH:-}"
apptainer exec --nv --overlay "$BASE/blackwell-torch.overlay":ro "$BASE/pytorch-cuda.sif" \
  /bb/bwvenv/bin/python "$BASE/scripts/inspect_moe_params.py"
echo done rc=$?
