#!/usr/bin/env bash
#SBATCH --job-name=bw1-dir
#SBATCH --partition=sixhour
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/bw1_dir_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/bw1_dir_%j.err

# Trainer lsgd via srun DIRECTO (sin torchrun/elastic): una IMA se propaga al
# hilo principal y con CUDA_LAUNCH_BLOCKING da traceback Python con linea.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF="$BASE/pytorch-cuda.sif"
OVL="$BASE/blackwell-torch.overlay"
VENV=/bb/bwvenv
cd "$BASE"
srun --export=ALL -n 2 apptainer exec --nv --overlay "$OVL":ro "$SIF" env \
  PYTHONDONTWRITEBYTECODE=1 CUDA_LAUNCH_BLOCKING=1 \
  "$VENV/bin/python" "$BASE/scripts/train_mdlm_moe_lsgd.py" \
  --data_cache "$BASE/data/train_ids_v5.npy" \
  --data "$BASE/data/train_corpus_v5.jsonl" \
  --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
  --n_experts 8 --expert_k 2 --seq_len 768 \
  --batch_size 4 --grad_accum 4 --mask_p 0.15 \
  --sync_every 1 --clip_grad 1.0 \
  --lr 2e-4 --warmup 5 --max_steps 5 \
  --output "$BASE/outputs/bw1_dir" 2>&1 | tail -70
echo "bw1-dir done rc=$?"
