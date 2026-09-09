#!/usr/bin/env bash
#SBATCH --job-name=smoke-base
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/smoke_base_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/smoke_base_%j.err
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
apptainer exec --nv --overlay "$BASE/blackwell-torch.overlay":ro "$BASE/pytorch-cuda.sif" /bb/bwvenv/bin/python \
  "$BASE/scripts/smoke_l1.py" \
  --ckpt "$BASE/outputs/bw1v5/checkpoint-g6000" \
  --corpus "$BASE/data/l1/train_corpus_l1.jsonl" \
  --n 10 --n-gen 120 --steps 64 \
  --out "$BASE/data/l1/smoke_base_report.json"
echo "smoke-base done rc=$?"
