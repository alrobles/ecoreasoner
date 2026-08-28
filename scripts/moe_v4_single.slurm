#!/usr/bin/env bash
#SBATCH --job-name=moe-v4-single
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:q6000:3
#SBATCH --mem=60G
#SBATCH --time=05:50:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_v4_single_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_v4_single_%j.err

# MoE v4 — SWEET SPOT productivo: baseline MoE 4 exp, DDP INTRA-NODO (3 GPU, 1 nodo).
# Evidencia: el MoE-DDP funciona en 1 nodo (fix-test) pero NO multi-nodo (all-reduce bloquea).
# Dataset v4 (1.7B tokens), baseline con fix (balance loss + find_unused_parameters=False).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
DATA="$BASE/data/train_ids_v4.npy"
[ -f "$DATA" ] || { echo "FALTA train_ids_v4.npy"; exit 1; }

srun --export=ALL -n 3 apptainer exec --nv pytorch-cuda.sif \
  python3 scripts/train_mdlm_moe.py \
  --data_cache "$DATA" \
  --data "$BASE/data/train_corpus_v4.jsonl" \
  --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
  --n_experts 4 --expert_k 1 --seq_len 768 \
  --batch_size 1 --grad_accum 4 --mask_p 0.15 \
  --lr 2e-4 --warmup 800 --max_steps 6000 \
  --output outputs/moe_v4_single
echo "MoE v4 single-node done rc=$?"