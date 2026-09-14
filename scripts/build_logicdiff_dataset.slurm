#!/bin/bash
#SBATCH --job-name=logicdiff-data
#SBATCH --partition=sixhour
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/build_logicdiff_dataset_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/build_logicdiff_dataset_%j.err

BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
OVL=$BASE/blackwell-torch.overlay
TOK=/beegfs/a474r867/hf-cache/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07

apptainer exec --nv --overlay "$OVL":ro "$SIF" /bb/bwvenv/bin/python -u "$BASE/scripts/build_logicdiff_dataset.py" \
    --in "$BASE/data/skeleton/train_skeleton_train.jsonl" \
    --out "$BASE/data/logicdiff/role_train.pt" \
    --tokenizer "$TOK" \
    --max-len 768 \
    --limit 100
