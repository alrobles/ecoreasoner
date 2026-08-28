#!/usr/bin/env bash
#SBATCH --job-name=moe-bw-smoke
#SBATCH --partition=sixhour
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_bw_smoke_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_bw_smoke_%j.err

# Valida el MoE v4 en Blackwell (sm_120) con DDP world=2, el fix de probe, batch 4,
# pocos pasos (sin OOM, sin deadlock de experts-unused).
#
# FIX 2026-08-28: antes usabamos `srun -n 2 apptainer exec --overlay ...` -> los 2 ranks
# montaban el MISMO overlay ext3 en paralelo y apptainer tomaba lock de escritura -> el
# 2o rank moria (exit 255, "currently in use by another process"). Ahora montamos el
# overlay UNA sola vez (ntasks=1) y lanzamos DDP local con torchrun --nproc_per_node=2
# dentro del mismo contenedor. Venv limpio /bb/bwvenv con torch 2.7.1+cu128 (sm_120 OK).
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF="$BASE/pytorch-cuda.sif"
OVL="$BASE/blackwell-torch.overlay"
VENV="/bb/bwvenv"
cd "$BASE"
srun --export=ALL -n 1 apptainer exec --nv --overlay "$OVL" "$SIF" \
  "$VENV/bin/python" -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29512 \
  scripts/train_mdlm_moe.py \
  --data_cache "$BASE/data/train_ids_v4.npy" \
  --data "$BASE/data/train_corpus_v4.jsonl" \
  --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
  --n_experts 4 --expert_k 1 --seq_len 768 \
  --batch_size 4 --grad_accum 4 --mask_p 0.15 \
  --max_steps 30 --warmup 5 --lr 2e-4 \
  --output outputs/moe_bw_smoke
echo "Blackwell smoke done rc=$?"
