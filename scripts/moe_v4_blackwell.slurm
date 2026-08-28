#!/usr/bin/env bash
#SBATCH --job-name=moe-bw-v4
#SBATCH --partition=sixhour
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:pro6000:2
#SBATCH --mem=60G
#SBATCH --time=05:50:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/beegfs/a474r867/ecoreasoner/logs/moe_bw_v4_%j.out
#SBATCH --error=/beegfs/a474r867/ecoreasoner/logs/moe_bw_v4_%j.err

# MoE v4 en Blackwell (pro6000, 96GB VRAM) — escapando el techo de VRAM del Q6000.
# DDP world=2 (2 GPU pro6000, 1 nodo). Mismo sweet spot arquitectural (h1024 l16 ne4)
# pero batch MÁS GRANDE: 96GB permiten batch 2-4 sin OOM -> más expertos activos/iter
# -> ataca la raíz del problema de grad (experts inactivos) además del probe.
set -euo pipefail
BASE=/beegfs/a474r867/ecoreasoner
SIF="$BASE/pytorch-cuda.sif"
OVL="$BASE/blackwell-torch.overlay"
VENV="/bb/bwvenv"
cd "$BASE"
DATA="$BASE/data/train_ids_v4.npy"
[ -f "$DATA" ] || { echo "FALTA train_ids_v4.npy"; exit 1; }

srun --export=ALL -n 2 apptainer exec --nv --overlay "$OVL" "$SIF" \
  "$VENV/bin/python" scripts/train_mdlm_moe.py \
  --data_cache "$DATA" \
  --data "$BASE/data/train_corpus_v4.jsonl" \
  --hidden 1024 --layers 16 --heads 16 --ff_mult 4 \
  --n_experts 4 --expert_k 1 --seq_len 768 \
  --batch_size 4 --grad_accum 4 --mask_p 0.15 \
  --lr 2e-4 --warmup 800 --max_steps 6000 \
  --output outputs/moe_bw_v4
echo "MoE v4 Blackwell done rc=$?"