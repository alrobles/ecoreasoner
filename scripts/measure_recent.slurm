#!/usr/bin/env bash
#SBATCH --job-name=meas-recent
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/meas_recent_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/meas_recent_%j.err

# Mide candidatos eco 2020-2025 con PMCID y margen vs corpus v5.
python3 /beegfs/a474r867/ecoreasoner/scripts/measure_pool_recent.py
echo "meas-recent DONE rc=$?"
