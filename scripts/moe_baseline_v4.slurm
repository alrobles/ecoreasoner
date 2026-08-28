#!/usr/bin/env bash
#SBATCH --job-name=moe-baseline-v4
#SBATCH --partition=sixhour
#SBATCH --nodes=4
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:q6000:3
#SBATCH --mem=60G
#SBATCH --time=05:50:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_baseline_v4_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_baseline_v4_%j.err

# BASELINE MoE v4 — train_mdlm_moe.py (DEMOSTRADO funciona en DDP, 12 ranks).
# El expE2 se congela en DDP; este baseline terminó el dense de control (6000 steps).
# v4 MoE: hidden 1024, 16 layers, n_experts=4 top-1, sobre corpus v4 (1.7B tokens).
# DDP 12 ranks (4 nodos × 3 Q6000). batch 1 + grad_accum 4.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
DATA="$BASE/data/train_ids_v4.npy"
[ -f "$DATA" ] || { echo "FALTA train_ids_v4.npy"; exit 1; }

srun --export=ALL -n 12 apptainer exec --nv pytorch-cuda.sif \
  python3 scripts/train_mdlm_moe.py \
  --data_cache "$DATA" \
  --data "$BASE/data/train_corpus_v4.jsonl" \
  --hidden 1024 \
  --layers 16 \
  --heads 16 \
  --ff_mult 4 \
  --n_experts 4 \
  --expert_k 1 \
  --seq_len 768 \
  --batch_size 1 \
  --grad_accum 4 \
  --mask_p 0.15 \
  --lr 2e-4 \
  --warmup 800 \
  --max_steps 6000 \
  --output outputs/moe_baseline_v4
echo "BASELINE MoE v4 done rc=$?"