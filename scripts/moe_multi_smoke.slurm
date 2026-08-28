#!/usr/bin/env bash
#SBATCH --job-name=moe-multi-smoke
#SBATCH --partition=sixhour
#SBATCH --time=00:20:00
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:q6000:2
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_multi_smoke_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_multi_smoke_%j.err

# =============================================================================
# moe-multi-smoke — VALIDA el MoE v4 MULTI-NODO (2 nodos x 2 Q6000 = 4 ranks).
# Objetivo: probar que el fix del probe resuelve el all-reduce multi-nodo que
# cuelga 9 veces el 2026-08-27. Con el patch MASTER_ADDR (nodo rank0) correcto,
# el all-reduce inter-nodo deberia funcionar con batch 4/rank.
# Si funciona => podemos escalar "4 tarjetas en 4 nodos" con las Q6000 (lento
# pero mucha capacidad). Si cuelga => evidencia de que el fix no basta multi-nodo.
# Patron: srun -N 2 -n 4 apptainer exec SIF (sin overlay; Q6000 sm_86 OK torch 2.4.1).
# =============================================================================
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF=$BASE/pytorch-cuda.sif
cd "$BASE"

echo "=== [$(date)] nodos: $SLURM_JOB_NODELIST ==="
srun --export=ALL -N 2 -n 4 --ntasks-per-node=2 apptainer exec --nv "$SIF" \
  python3 scripts/train_mdlm_moe.py \
  --data_cache "$BASE/data/train_ids_v5.npy" \
  --data "$BASE/data/train_corpus_v5.jsonl" \
  --hidden 512 --layers 8 --heads 8 --ff_mult 4 \
  --n_experts 4 --expert_k 1 --seq_len 512 \
  --batch_size 4 --grad_accum 4 --mask_p 0.15 \
  --max_steps 30 --warmup 5 --lr 2e-4 \
  --output outputs/moe_multi_smoke
echo "=== [$(date)] multi smoke done rc=$? ==="