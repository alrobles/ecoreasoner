#!/bin/bash
#SBATCH --job-name=probe-sft-mt
#SBATCH --partition=sixhour
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|q6000|a100|l40|a40|q8000|pro6000"
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/probe_sft_mt_%j.out
# Sonda multi-turno O5 sobre sft-v4-g16000 (probe_sft_mt.py).
# Sin mi210 (ROCm); incluye v100/q6000 idle — el modelo son ~4.5GB fp32.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
export PYTHONPATH=/beegfs/a474r867/pydeps_tf451${PYTHONPATH:+:$PYTHONPATH}
export CKPT=${CKPT:-$BASE/runs/sft-v4/checkpoint-g16000/ema_model.pt}
export TOK=${TOK:-/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "?")
echo "probe-sft-mt ckpt=$CKPT gpu=$GPU_NAME"
apptainer exec --nv "$SIF" python3 "$BASE/scripts/probe_sft_mt.py"
