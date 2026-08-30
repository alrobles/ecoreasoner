#!/usr/bin/env bash
#SBATCH --job-name=probe-moe-lg
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/probe_moe_logits_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/probe_moe_logits_%j.err
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
export PYTHONPATH="$BASE/venv-extra:${PYTHONPATH:-}"
apptainer exec --nv --overlay "$BASE/blackwell-torch.overlay":ro "$BASE/pytorch-cuda.sif" \
  /bb/bwvenv/bin/python "$BASE/scripts/probe_moe_logits.py"
echo "probe done rc=$?"