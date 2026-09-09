#!/bin/bash
#SBATCH --job-name=measure-pool
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/measure_pool_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/measure_pool_%j.err
set -euo pipefail
cd /beegfs/a474r867/ecoreasoner
python3 scripts/measure_pool.py
echo "[$(date)] DONE"