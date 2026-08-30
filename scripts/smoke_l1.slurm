#!/usr/bin/env bash
#SBATCH --job-name=smoke-l1
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/smoke_l1_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/smoke_l1_%j.err

# Smoke L1: el dLLM (bw2/checkpoint-g7500) genera tool calls JSON validas
# ante prompts [INSTRUCCION]+[CONTEXTO]+[ACCION] (denoising LLaDA-style).
# Corre DENTRO del contenedor con el venv del entrenamiento (torch cu128 para
# Blackwell sm_120) via overlay :ro (solo lectura, el training ya termino).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
OVL="$BASE/blackwell-torch.overlay"
SIF="$BASE/pytorch-cuda.sif"
apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python \
  "$BASE/scripts/smoke_l1.py" \
  --ckpt "$BASE/outputs/bw2/checkpoint-g7500" \
  --corpus "$BASE/data/l1/train_corpus_l1.jsonl" \
  --n 40 --n-gen 120 --steps 64 \
  --out "$BASE/data/l1/smoke_l1_report.json"
echo "smoke-l1 done rc=$?"