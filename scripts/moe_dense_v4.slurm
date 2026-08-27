#!/usr/bin/env bash
#SBATCH --job-name=moe-dense-v4
#SBATCH --partition=sixhour
#SBATCH --nodes=3
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:q6000:4
#SBATCH --mem=60G
#SBATCH --time=05:50:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_dense_v4_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_dense_v4_%j.err

# CONTROL DENSE v4 — baseline para comparar contra el MoE experimental E2+E4 v4.
# Mismo hidden 1024 / layers 16 / heads 16 / ff_mult 4 que el MoE v4, PERO
# n_experts=1 → FFN denso (todo activo). Permite juzgar si el MoE (2B/595M act)
# aporta valor sobre un dense de coste comparable, sobre el MISMO dataset v4.
# DDP 12 ranks (3 nodos × 4 Q6000 = 12 GPU).
# REQUISITO: train_ids_v4.npy exista (pre-tokenize v4 completo).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
cd "$BASE"
DATA="$BASE/data/train_ids_v4.npy"
[ -f "$DATA" ] || { echo "FALTA train_ids_v4.npy — pre-tokenize v4 no completado"; exit 1; }

srun --export=ALL -n 12 apptainer exec --nv pytorch-cuda.sif \
  python3 scripts/train_mdlm_moe.py \
  --data_cache "$DATA" \
  --data "$BASE/data/train_corpus_v4.jsonl" \
  --hidden 1024 \
  --layers 16 \
  --heads 16 \
  --ff_mult 4 \
  --n_experts 1 \
  --expert_k 1 \
  --seq_len 768 \
  --batch_size 1 \
  --grad_accum 4 \
  --mask_p 0.15 \
  --lr 2e-4 \
  --warmup 800 \
  --max_steps 6000 \
  --output outputs/moe_dense_v4
echo "DENSE v4 done rc=$?"