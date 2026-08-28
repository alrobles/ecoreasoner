#!/usr/bin/env bash
#SBATCH --job-name=moe-bw-smoke
#SBATCH --partition=sixhour
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=40G
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_bw_smoke_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_bw_smoke_%j.err

# Valida el MoE v4 en Blackwell (sm_120) con DDP world=2, el fix de probe, batch 4,
# pocos pasos (sin OOM, sin deadlock de experts-unused). Usa el overlay con torch 2.7.1
# cu128 (setup_blackwell_torch.slurm) y el venv limpio /bb/bwvenv.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF="$BASE/pytorch-cuda.sif"
OVL="$BASE/blackwell-torch.overlay"
VENV="/bb/bwvenv"
cd "$BASE"
srun --export=ALL -n 2 apptainer exec --nv --overlay "$OVL" "$SIF" \
  "$VENV/bin/python" scripts/train_mdlm_moe.py \
  --data_cache "$BASE/data/train_ids_v4.npy" \
  --data "$BASE/data/train_corpus_v4.jsonl" \
  --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
  --n_experts 4 --expert_k 1 --seq_len 768 \
  --batch_size 4 --grad_accum 4 --mask_p 0.15 \
  --max_steps 30 --warmup 5 --lr 2e-4 \
  --output outputs/moe_bw_smoke
echo "Blackwell smoke done rc=$?"