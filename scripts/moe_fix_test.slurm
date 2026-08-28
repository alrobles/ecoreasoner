#!/usr/bin/env bash
#SBATCH --job-name=moe-fix-test
#SBATCH --partition=sixhour
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:q6000:1
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_fix_test_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_fix_test_%j.err
# Test del fix de deadlock: baseline MoE 4exp, 1 GPU, pocos steps, single-GPU (sin DDP).
# Verifica que el loss aux de balance no rompe y que produce steps.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
srun --export=ALL -n 1 apptainer exec --nv pytorch-cuda.sif \
  python3 scripts/train_mdlm_moe.py \
  --data_cache "$BASE/data/train_ids_v4.npy" \
  --data "$BASE/data/train_corpus_v4.jsonl" \
  --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
  --n_experts 4 --expert_k 1 --seq_len 768 --batch_size 1 --grad_accum 2 \
  --max_steps 50 --warmup 10 --lr 2e-4 \
  --output outputs/moe_fix_test
echo "fix-test done rc=$?"