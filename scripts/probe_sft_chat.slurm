#!/bin/bash
#SBATCH --job-name=probe-sft-chat
#SBATCH --partition=sixhour
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/probe_sft_chat_%j.out
# Probe de chat sobre ckpt SFT-v2 (probe_sft_chat.py). PROMPTS via --export.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
export PYTHONPATH=/beegfs/a474r867/pydeps_tf451${PYTHONPATH:+:$PYTHONPATH}
export CKPT=${CKPT:-$BASE/runs/sft-v2/checkpoint-g35500/ema_model.pt}
export TOK=${TOK:-/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "?")
echo "probe-sft ckpt=$CKPT gpu=$GPU_NAME"
apptainer exec --nv "$SIF" python3 "$BASE/scripts/probe_sft_chat.py"
