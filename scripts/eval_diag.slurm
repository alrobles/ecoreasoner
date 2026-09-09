#!/bin/bash
#SBATCH --partition=sixhour
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --gres=gpu:mi210:1
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/eval_diag-%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/eval_diag-%j.err
#SBATCH --job-name=eco-eval-diag
set -u
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/qwen35-rocm-v2.sif
echo "=== diag $(date -u) host=$(hostname) ==="
echo "--- python3 version ---"
apptainer exec --nv $SIF python3 --version 2>&1
echo "--- python3 check torch ---"
apptainer exec --nv $SIF python3 -c "import sys; print('ver', sys.version)" 2>&1 | head -5
echo "--- list eval script ---"
ls -la /beegfs/a474r867/ecoreasoner/scripts/eval_baseline_A.py 2>&1
echo "--- run eval with --help (argparse exit 0) ---"
apptainer exec --nv $SIF python3 /beegfs/a474r867/ecoreasoner/scripts/eval_baseline_A.py --help 2>&1 | head -20
echo "=== diag fin ==="